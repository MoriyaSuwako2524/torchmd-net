# Copyright Universitat Pompeu Fabra 2020-2023  https://www.compscience.org
# Distributed under the MIT License.
# (See accompanying file README.md file or copy at http://opensource.org/licenses/MIT)
from glob import glob
import os
import re
import tempfile
from typing import Optional, List, Tuple, Dict
import torch
from torch.autograd import grad
from torch import nn, Tensor
from torchmdnet.models import output_modules
from torchmdnet.models.wrappers import AtomFilter
from torchmdnet.models.utils import dtype_mapping
from torchmdnet import priors
from lightning_utilities.core.rank_zero import rank_zero_warn
import warnings
import zipfile


def create_model(args, prior_model=None, mean=None, std=None):
    """Create a model from the given arguments.

    Run `torchmd-train --help` for a description of the arguments.

    Args:
        args (dict): Arguments for the model.
        prior_model (nn.Module, optional): Prior model to use. Defaults to None.
        mean (torch.Tensor, optional): Mean of the training data. Defaults to None.
        std (torch.Tensor, optional): Standard deviation of the training data. Defaults to None.

    Returns:
        nn.Module: An instance of the TorchMD_Net model.
    """
    dtype = dtype_mapping[args["precision"]]
    if "box_vecs" not in args:
        args["box_vecs"] = None
    if "check_errors" not in args:
        args["check_errors"] = True
    if "static_shapes" not in args:
        args["static_shapes"] = False
    if "vector_cutoff" not in args:
        args["vector_cutoff"] = False

    shared_args = dict(
        hidden_channels=args["embedding_dimension"],
        num_layers=args["num_layers"],
        num_rbf=args["num_rbf"],
        rbf_type=args["rbf_type"],
        trainable_rbf=args["trainable_rbf"],
        activation=args["activation"],
        cutoff_lower=float(args["cutoff_lower"]),
        cutoff_upper=float(args["cutoff_upper"]),
        max_z=args["max_z"],
        check_errors=bool(args["check_errors"]),
        max_num_neighbors=args["max_num_neighbors"],
        box_vecs=(
            torch.tensor(args["box_vecs"], dtype=dtype)
            if args["box_vecs"] is not None
            else None
        ),
        dtype=dtype,
    )

    # representation network
    if args["model"] == "graph-network":
        from torchmdnet.models.torchmd_gn import TorchMD_GN

        is_equivariant = False
        representation_model = TorchMD_GN(
            num_filters=args["embedding_dimension"],
            aggr=args["aggr"],
            neighbor_embedding=args["neighbor_embedding"],
            **shared_args,
        )
    elif args["model"] == "transformer":
        from torchmdnet.models.torchmd_t import TorchMD_T

        is_equivariant = False
        representation_model = TorchMD_T(
            attn_activation=args["attn_activation"],
            num_heads=args["num_heads"],
            distance_influence=args["distance_influence"],
            neighbor_embedding=args["neighbor_embedding"],
            **shared_args,
        )
    elif args["model"] == "equivariant-transformer":
        from torchmdnet.models.torchmd_et import TorchMD_ET

        is_equivariant = True
        representation_model = TorchMD_ET(
            attn_activation=args["attn_activation"],
            num_heads=args["num_heads"],
            distance_influence=args["distance_influence"],
            neighbor_embedding=args["neighbor_embedding"],
            vector_cutoff=args["vector_cutoff"],
            **shared_args,
        )
    elif args["model"] == "tensornet":
        from torchmdnet.models.tensornet import TensorNet

        # Setting is_equivariant to False to enforce the use of Scalar output module instead of EquivariantScalar
        is_equivariant = False
        representation_model = TensorNet(
            equivariance_invariance_group=args["equivariance_invariance_group"],
            static_shapes=args["static_shapes"],
            **shared_args,
        )
    else:
        raise ValueError(f'Unknown architecture: {args["model"]}')

    # atom filter
    if not args["derivative"] and args["atom_filter"] > -1:
        representation_model = AtomFilter(representation_model, args["atom_filter"])
    elif args["atom_filter"] > -1:
        raise ValueError("Derivative and atom filter can't be used together")

    # prior model
    if args["prior_model"] and prior_model is None:
        # instantiate prior model if it was not passed to create_model (i.e. when loading a model)
        prior_model = create_prior_models(args)

    # create output network
    output_prefix = "Equivariant" if is_equivariant else ""
    output_model = getattr(output_modules, output_prefix + args["output_model"])(
        args["embedding_dimension"],
        activation=args["activation"],
        reduce_op=args["reduce_op"],
        dtype=dtype,
        num_hidden_layers=args.get("output_mlp_num_layers", 0),
    )

    # ==========================================================
    # Dynamic multi-output construction
    # ==========================================================

    pred_dict = args.get("pred_dict", {"y": 1.0, "neg_dy": 1.0})
    output_modules_dict = {}

    # always include main scalar output
    output_modules_dict["y"] = getattr(output_modules, output_prefix + args["output_model"])(
        args["embedding_dimension"],
        activation=args["activation"],
        reduce_op=args["reduce_op"],
        dtype=dtype,
        num_hidden_layers=args.get("output_mlp_num_layers", 0),
    )

    # add extra requested outputs
    if "vec" in pred_dict:
        output_modules_dict["vec"] = output_modules.DipoleMoment(
            args["embedding_dimension"],
            activation=args["activation"],
            reduce_op=args["reduce_op"],
            dtype=dtype,
            num_hidden_layers=args.get("output_mlp_num_layers", 0),
        )

    if "charge" in pred_dict:
        from torchmdnet.models.output_modules import AtomicCharge

        reduce_op_charge = "none"
        output_modules_dict["charge"] = AtomicCharge(
            args["embedding_dimension"],
            activation=args["activation"],
            reduce_op=args["reduce_op"],
            dtype=dtype,
            num_hidden_layers=args.get("output_mlp_num_layers", 0),
        )



    # extend here for more (polar, esp, etc.)
    # ==========================================================
    print(f"output_modules_dict:{output_modules_dict}")
    model = TorchMD_Net_MultiOutput(
        representation_model,
        output_modules_dict,
        prior_model=prior_model,
        mean=mean,
        std=std,
        derivative=args["derivative"],
        dtype=dtype,
    )

    return model



def load_ensemble(filepath, args=None, device="cpu", return_std=False, **kwargs):
    """Load an ensemble of models from a list of checkpoint files or a zip file.

    Args:
        filepath (str or list): Can be any of the following:

            - Path to a zip file containing multiple checkpoint files.
            - List of paths to checkpoint files.

        args (dict, optional): Arguments for the model. Defaults to None.
        device (str, optional): Device on which the model should be loaded. Defaults to "cpu".
        return_std (bool, optional): Whether to return the standard deviation of the predictions. Defaults to False.
        **kwargs: Extra keyword arguments for the model, will be passed to :py:mod:`load_model`.

    Returns:
        nn.Module: An instance of :py:mod:`Ensemble`.
    """
    if isinstance(filepath, (list, tuple)):
        assert all(isinstance(f, str) for f in filepath), "Invalid filepath list."
        model_list = [
            load_model(f, args=args, device=device, **kwargs) for f in filepath
        ]
    elif filepath.endswith(".zip"):
        with tempfile.TemporaryDirectory() as tmpdir:
            with zipfile.ZipFile(filepath, "r") as z:
                z.extractall(tmpdir)
            ckpt_list = glob(os.path.join(tmpdir, "*.ckpt"))
            assert len(ckpt_list) > 0, "No checkpoint files found in zip file."
            model_list = [
                load_model(f, args=args, device=device, **kwargs) for f in ckpt_list
            ]
    else:
        raise ValueError(
            "Invalid filepath. Must be a list of paths or a path to a zip file."
        )
    return Ensemble(
        model_list,
        return_std=return_std,
    )



def load_model(filepath, args=None, device="cpu", return_std=False, **kwargs):
    """Load a model from a checkpoint file.

       If a list of paths or a path to a zip file is given, an :py:mod:`Ensemble` model is returned.
    Args:
        filepath (str or list): Can be any of the following:

            - Path to a checkpoint file. In this case, a :py:mod:`TorchMD_Net` model is returned.
            - Path to a zip file containing multiple checkpoint files. In this case, an :py:mod:`Ensemble` model is returned.
            - List of paths to checkpoint files. In this case, an :py:mod:`Ensemble` model is returned.

        args (dict, optional): Arguments for the model. Defaults to None.
        device (str, optional): Device on which the model should be loaded. Defaults to "cpu".
        return_std (bool, optional): Whether to return the standard deviation of an Ensemble model. Defaults to False.
        **kwargs: Extra keyword arguments for the model.

    Returns:
        nn.Module: An instance of the TorchMD_Net model or an Ensemble model.
    """
    isEnsemble = isinstance(filepath, (list, tuple)) or filepath.endswith(".zip")
    if isEnsemble:
        return load_ensemble(
            filepath, args=args, device=device, return_std=return_std, **kwargs
        )
    assert isinstance(filepath, str)
    ckpt = torch.load(filepath, map_location="cpu", weights_only=False)
    if args is None:
        args = ckpt["hyper_parameters"]

    delta_learning = args["remove_ref_energy"] if "remove_ref_energy" in args else False

    for key, value in kwargs.items():
        if not key in args:
            warnings.warn(f"Unknown hyperparameter: {key}={value}")
        args[key] = value

    model = create_model(args)
    if delta_learning and "remove_ref_energy" in kwargs:
        if not kwargs["remove_ref_energy"]:
            assert (
                len(model.prior_model) > 0
            ), "Atomref prior must be added during training (with enable=False) for total energy prediction."
            assert isinstance(
                model.prior_model[-1], priors.Atomref
            ), "I expected the last prior to be Atomref."
            # Set the Atomref prior to enabled
            model.prior_model[-1].enable = True

    state_dict = {re.sub(r"^model\.", "", k): v for k, v in ckpt["state_dict"].items()}
    # In ET, before we had output_model.output_network.{0,1}.update_net.[0-9].{weight,bias}
    # Now we have output_model.output_network.{0,1}.update_net.layers.[0-9].{weight,bias}
    # In other models, we had output_model.output_network.{0,1}.{weight,bias},
    # which is now output_model.output_network.layers.{0,1}.{weight,bias}
    # This change was introduced in https://github.com/torchmd/torchmd-net/pull/314
    patterns = [
        (
            r"output_model.output_network.(\d+).update_net.(\d+).",
            r"output_model.output_network.\1.update_net.layers.\2.",
        ),
        (
            r"output_model.output_network.([02]).(weight|bias)",
            r"output_model.output_network.layers.\1.\2",
        ),
    ]
    for p in patterns:
        state_dict = {re.sub(p[0], p[1], k): v for k, v in state_dict.items()}

    model.load_state_dict(state_dict)
    return model.to(device)




def create_prior_models(args, dataset=None):
    """Parse the prior_model configuration option and create the prior models.

    The information can be passed in different ways via the args dictionary, which must contain at least the key "prior_model".

    1. A single prior model name and its arguments as a dictionary:

    .. code:: python

      args = {
          "prior_model": "Atomref",
          "prior_args": {"max_z": 100}
      }


    2. A list of prior model names and their arguments as a list of dictionaries:

    .. code:: python

      args = {
          "prior_model": ["Atomref", "D2"],
          "prior_args": [{"max_z": 100}, {"max_z": 100}]
      }


    3. A list of prior model names and their arguments as a dictionary:

    .. code:: python

      args = {
          "prior_model": [{"Atomref": {"max_z": 100}}, {"D2": {"max_z": 100}}]
      }


    Args:
        args (dict): Arguments for the model.
        dataset (torch_geometric.data.Dataset, optional): A dataset from which to extract the atomref values. Defaults to None.

    Returns:
        list: A list of prior models.

    """
    prior_models = []
    if args["prior_model"]:
        prior_model = args["prior_model"]
        prior_names = []
        prior_args = []
        if not isinstance(prior_model, list):
            prior_model = [prior_model]
        for prior in prior_model:
            if isinstance(prior, dict):
                for key, value in prior.items():
                    prior_names.append(key)
                    if value is None:
                        prior_args.append({})
                    else:
                        prior_args.append(value)
            else:
                prior_names.append(prior)
                prior_args.append({})
        if "prior_args" in args and args["prior_args"] is not None:
            prior_args = args["prior_args"]
            if not isinstance(prior_args, list):
                prior_args = [prior_args]
        for name, arg in zip(prior_names, prior_args):
            assert hasattr(priors, name), (
                f"Unknown prior model {name}. "
                f"Available models are {', '.join(priors.__all__)}"
            )
            # initialize the prior model
            prior_models.append(getattr(priors, name)(dataset=dataset, **arg))
    return prior_models


class TorchMD_Net(nn.Module):
    """The main TorchMD-Net model.

    The TorchMD_Net class combines a given representation model (such as the equivariant transformer),
    an output model (such as the scalar output module), and a prior model (such as the atomref prior).
    It produces a Module that takes as input a series of atom features and outputs a scalar value
    (i.e., energy for each batch/molecule). If `derivative` is True, it also outputs the negative of
    its derivative with respect to the positions (i.e., forces for each atom).

    Parameters
    ----------
    representation_model : nn.Module
        A model that takes as input the atomic numbers, positions, batch indices, and optionally
        charges and spins. It must return a tuple of the form (x, v, z, pos, batch), where x
        are the atom features, v are the vector features (if any), z are the atomic numbers,
        pos are the positions, and batch are the batch indices. See TorchMD_ET for more details.
    output_model : nn.Module
        A model that takes as input the atom features, vector features (if any), atomic numbers,
        positions, and batch indices. See OutputModel for more details.
    prior_model : nn.Module, optional
        A model that takes as input the atom features, atomic numbers, positions, and batch
        indices. See BasePrior for more details. Defaults to None.
    mean : torch.Tensor, optional
        Mean of the training data. Defaults to None.
    std : torch.Tensor, optional
        Standard deviation of the training data. Defaults to None.
    derivative : bool, optional
        Whether to compute the derivative of the outputs via backpropagation. Defaults to False.
    dtype : torch.dtype, optional
        Data type of the model. Defaults to torch.float32.

    """

    def __init__(
        self,
        representation_model,
        output_model,
        prior_model=None,
        mean=None,
        std=None,
        derivative=False,
        dtype=torch.float32,
    ):
        super(TorchMD_Net, self).__init__()
        self.representation_model = representation_model.to(dtype=dtype)
        self.output_model = output_model.to(dtype=dtype)

        if not output_model.allow_prior_model and prior_model is not None:
            prior_model = None
            rank_zero_warn(
                (
                    "Prior model was given but the output model does "
                    "not allow prior models. Dropping the prior model."
                )
            )
        if isinstance(prior_model, priors.base.BasePrior):
            prior_model = [prior_model]
        self.prior_model = (
            None
            if prior_model is None
            else torch.nn.ModuleList(prior_model).to(dtype=dtype)
        )

        self.derivative = derivative

        mean = torch.scalar_tensor(0) if mean is None else mean
        self.register_buffer("mean", mean.to(dtype=dtype))
        std = torch.scalar_tensor(1) if std is None else std
        self.register_buffer("std", std.to(dtype=dtype))

        self.reset_parameters()

    def reset_parameters(self):
        self.representation_model.reset_parameters()
        self.output_model.reset_parameters()
        if self.prior_model is not None:
            for prior in self.prior_model:
                prior.reset_parameters()

    def forward(
        self,
        z: Tensor,
        pos: Tensor,
        batch: Optional[Tensor] = None,
        box: Optional[Tensor] = None,
        q: Optional[Tensor] = None,
        s: Optional[Tensor] = None,
        extra_args: Optional[Dict[str, Tensor]] = None,
    ) -> Tuple[Tensor, Tensor]:
        """
        Compute the output of the model.

        This function optionally supports periodic boundary conditions with
        arbitrary triclinic boxes.  The box vectors `a`, `b`, and `c` must satisfy
        certain requirements:

        .. code:: python

           a[1] = a[2] = b[2] = 0
           a[0] >= 2*cutoff, b[1] >= 2*cutoff, c[2] >= 2*cutoff
           a[0] >= 2*b[0]
           a[0] >= 2*c[0]
           b[1] >= 2*c[1]


        These requirements correspond to a particular rotation of the system and
        reduced form of the vectors, as well as the requirement that the cutoff be
        no larger than half the box width.

        Args:
            z (Tensor): Atomic numbers of the atoms in the molecule. Shape: (N,).
            pos (Tensor): Atomic positions in the molecule. Shape: (N, 3).
            batch (Tensor, optional): Batch indices for the atoms in the molecule. Shape: (N,).
            box (Tensor, optional): Box vectors. Shape (3, 3).
            The vectors defining the periodic box.  This must have shape `(3, 3)`,
            where `box_vectors[0] = a`, `box_vectors[1] = b`, and `box_vectors[2] = c`.
            If this is omitted, periodic boundary conditions are not applied.
            q (Tensor, optional): Atomic charges in the molecule. Shape: (N,).
            s (Tensor, optional): Atomic spins in the molecule. Shape: (N,).
            extra_args (Dict[str, Tensor], optional): Extra arguments to pass to the prior model.

        Returns:
            Tuple[Tensor, Optional[Tensor]]: The output of the model and the derivative of the output with respect to the positions if derivative is True, None otherwise.
        """
        assert z.dim() == 1 and z.dtype == torch.long
        batch = torch.zeros_like(z) if batch is None else batch

        if self.derivative:
            pos.requires_grad_(True)
        # run the potentially wrapped representation model
        x, v, z, pos, batch = self.representation_model(
            z, pos, batch, box=box, q=q, s=s
        )
        # apply the output network
        x = self.output_model.pre_reduce(x, v, z, pos, batch)

        # scale by data standard deviation
        if self.std is not None:
            x = x * self.std

        # apply atom-wise prior model
        if self.prior_model is not None:
            for prior in self.prior_model:
                x = prior.pre_reduce(x, z, pos, batch, extra_args)

        # aggregate atoms
        x = self.output_model.reduce(x, batch)

        # shift by data mean
        if self.mean is not None:
            x = x + self.mean

        # apply output model after reduction
        y = self.output_model.post_reduce(x)

        # apply molecular-wise prior model
        if self.prior_model is not None:
            for prior in self.prior_model:
                y = prior.post_reduce(y, z, pos, batch, box, extra_args)
        # compute gradients with respect to coordinates

        if self.derivative:
            grad_outputs: List[Optional[torch.Tensor]] = [torch.ones_like(y)]
            dy = grad(
                [y],
                [pos],
                grad_outputs=grad_outputs,
                create_graph=self.training,
                retain_graph=self.training,
            )[0]
            assert dy is not None, "Autograd returned None for the force prediction."
            return y, -dy
        # Returning an empty tensor allows to decorate this method as always returning two tensors.
        # This is required to overcome a TorchScript limitation, xref https://github.com/openmm/openmm-torch/issues/135
        return y, torch.empty(0)


class TorchMD_Net_Dipole(nn.Module):
    """TorchMD-Net variant with dipole moment prediction.

    Same structure as TorchMD_Net, but returns an additional vector output 'vec'
    corresponding to the molecular dipole moment.
    """

    def __init__(
        self,
        representation_model,
        output_model,
        dipole_output,
        prior_model=None,
        mean=None,
        std=None,
        derivative=False,
        dtype=torch.float32,
    ):
        super(TorchMD_Net_Dipole, self).__init__()
        self.representation_model = representation_model.to(dtype=dtype)
        self.output_model = output_model.to(dtype=dtype)
        self.dipole_output = dipole_output.to(dtype=dtype)

        if not output_model.allow_prior_model and prior_model is not None:
            prior_model = None
            rank_zero_warn(
                (
                    "Prior model was given but the output model does "
                    "not allow prior models. Dropping the prior model."
                )
            )
        if isinstance(prior_model, priors.base.BasePrior):
            prior_model = [prior_model]
        self.prior_model = (
            None
            if prior_model is None
            else torch.nn.ModuleList(prior_model).to(dtype=dtype)
        )

        self.derivative = derivative
        mean = torch.scalar_tensor(0) if mean is None else mean
        self.register_buffer("mean", mean.to(dtype=dtype))
        std = torch.scalar_tensor(1) if std is None else std
        self.register_buffer("std", std.to(dtype=dtype))

        self.reset_parameters()

    def reset_parameters(self):
        self.representation_model.reset_parameters()
        self.output_model.reset_parameters()
        self.dipole_output.reset_parameters()
        if self.prior_model is not None:
            for prior in self.prior_model:
                prior.reset_parameters()

    def forward(
        self,
        z: Tensor,
        pos: Tensor,
        batch: Optional[Tensor] = None,
        box: Optional[Tensor] = None,
        q: Optional[Tensor] = None,
        s: Optional[Tensor] = None,
        extra_args: Optional[Dict[str, Tensor]] = None,
    ) -> Dict[str, Tensor]:
        assert z.dim() == 1 and z.dtype == torch.long
        batch = torch.zeros_like(z) if batch is None else batch

        if self.derivative:
            pos.requires_grad_(True)

        # representation
        x, v, z, pos, batch = self.representation_model(
            z, pos, batch, box=box, q=q, s=s
        )

        # main energy output
        x_e = self.output_model.pre_reduce(x, v, z, pos, batch)
        if self.std is not None:
            x_e = x_e * self.std
        if self.prior_model is not None:
            for prior in self.prior_model:
                x_e = prior.pre_reduce(x_e, z, pos, batch, extra_args)
        x_e = self.output_model.reduce(x_e, batch)
        if self.mean is not None:
            x_e = x_e + self.mean
        y = self.output_model.post_reduce(x_e)
        if self.prior_model is not None:
            for prior in self.prior_model:
                y = prior.post_reduce(y, z, pos, batch, box, extra_args)

        # compute forces
        if self.derivative:
            grad_outputs: List[Optional[torch.Tensor]] = [torch.ones_like(y)]
            dy = grad(
                [y],
                [pos],
                grad_outputs=grad_outputs,
                create_graph=self.training,
                retain_graph=self.training,
            )[0]
        else:
            dy = torch.empty(0, device=pos.device)

        # dipole output (vector)
        x_d = self.dipole_output.pre_reduce(x, v, z, pos, batch)
        vec = self.dipole_output.reduce(x_d, batch)

        return {"y": y, "neg_dy": -dy, "vec": vec}





class Ensemble(torch.nn.ModuleList):
    """Average predictions over an ensemble of TorchMD-Net models.

       This module behaves like a single TorchMD-Net model, but its forward method returns the average and standard deviation of the predictions over all models it was initialized with.

    Args:
        modules (List[nn.Module]): List of :py:mod:`TorchMD_Net` models to average predictions over.
        return_std (bool, optional): Whether to return the standard deviation of the predictions. Defaults to False. If set to True, the model returns 4 arguments (mean_y, mean_neg_dy, std_y, std_neg_dy) instead of 2 (mean_y, mean_neg_dy).
    """

    def __init__(self, modules: List[nn.Module], return_std: bool = False):
        for module in modules:
            assert isinstance(module, TorchMD_Net)
        super().__init__(modules)
        self.return_std = return_std

    def forward(
        self,
        *args,
        **kwargs,
    ):
        """Average predictions over all models in the ensemble.
        The arguments to this function are simply relayed to the forward method of each :py:mod:`TorchMD_Net` model in the ensemble.
        Args:
            *args: Positional arguments to forward to the models.
            **kwargs: Keyword arguments to forward to the models.
        Returns:
            Tuple[Tensor, Optional[Tensor]] or Tuple[Tensor, Optional[Tensor], Tensor, Optional[Tensor]]: The average and standard deviation of the predictions over all models in the ensemble. If return_std is False, the output is a tuple (mean_y, mean_neg_dy). If return_std is True, the output is a tuple (mean_y, mean_neg_dy, std_y, std_neg_dy).

        """
        y = []
        neg_dy = []
        for model in self:
            res = model(*args, **kwargs)
            y.append(res[0])
            neg_dy.append(res[1])
        y = torch.stack(y)
        neg_dy = torch.stack(neg_dy)
        y_mean = torch.mean(y, axis=0)
        neg_dy_mean = torch.mean(neg_dy, axis=0)
        y_std = torch.std(y, axis=0)
        neg_dy_std = torch.std(neg_dy, axis=0)

        if self.return_std:
            return y_mean, neg_dy_mean, y_std, neg_dy_std
        else:
            return y_mean, neg_dy_mean



import torch
from torch import nn
from torch.autograd import grad


class TorchMD_Net_MultiOutput(nn.Module):
    """
    TorchMD-Net variant that supports multiple parallel output heads
    with optional mean/std normalization per head.

    Each output head can have its own mean/std (from DataModule._standardize()).
    """

    def __init__(
        self,
        representation_model,
        output_modules: dict[str, nn.Module],
        prior_model=None,
        mean=None,
        std=None,
        derivative=False,
        dtype=torch.float32,
    ):
        super().__init__()
        self.representation_model = representation_model.to(dtype=dtype)
        self.output_modules = nn.ModuleDict(output_modules)
        self.prior_model = (
            None
            if prior_model is None
            else torch.nn.ModuleList(prior_model).to(dtype=dtype)
        )
        self.derivative = derivative

        if mean is None:
            mean = torch.scalar_tensor(0.0, dtype=dtype)
        if std is None:
            std = torch.scalar_tensor(1.0, dtype=dtype)

        if isinstance(mean, dict):
            for k, v in mean.items():
                self.register_buffer(f"mean_{k}", v.to(dtype=dtype))
        else:
            self.register_buffer("mean", mean.to(dtype=dtype))

        if isinstance(std, dict):
            for k, v in std.items():
                self.register_buffer(f"std_{k}", v.to(dtype=dtype))
        else:
            self.register_buffer("std", std.to(dtype=dtype))
            


    def get_mean_std(self, key):
        """Return (mean, std) for a given output key."""
        mean_name = f"mean_{key}"
        std_name = f"std_{key}"
        mean = getattr(self, mean_name) if hasattr(self, mean_name) else getattr(self, "mean")
        std = getattr(self, std_name) if hasattr(self, std_name) else getattr(self, "std")
        return mean, std

    def forward(self, z, pos, batch=None, box=None, q=None, s=None, extra_args=None):
        batch = torch.zeros_like(z) if batch is None else batch
        if self.derivative:
            pos.requires_grad_(True)

        x, v, z, pos, batch = self.representation_model(z, pos, batch, box=box, q=q, s=s)
        results = {}

        for key, outmod in self.output_modules.items():
            mean, std = self.get_mean_std(key)
            x_out = outmod.pre_reduce(x, v, z, pos, batch)

            x_out = x_out * std

            if self.prior_model is not None:
                for prior in self.prior_model:
                    x_out = prior.pre_reduce(x_out, z, pos, batch, extra_args)

            x_out = outmod.reduce(x_out, batch)
            x_out = x_out + mean 

            y = outmod.post_reduce(x_out)

            if self.prior_model is not None:
                for prior in self.prior_model:
                    y = prior.post_reduce(y, z, pos, batch, box, extra_args)

            results[key] = y

        if self.derivative and "y" in results:
            grad_outputs = [torch.ones_like(results["y"])]
            dy = grad(
                [results["y"]],
                [pos],
                grad_outputs=grad_outputs,
                create_graph=self.training,
                retain_graph=self.training,
            )[0]
            results["neg_dy"] = -dy

        return results
    def load_state_dict(self, state_dict, strict=True, assign=False,dtype=torch.float32):
        print(f"Load state dict output modules dict:{self.output_modules.items()}")
        for key, outmod in self.output_modules.items():
            
            mean_name = f"mean_{key}"
            std_name = f"std_{key}"
            
            self.register_buffer(f"mean_{key}", state_dict[mean_name].to(dtype=dtype))
            self.register_buffer(f"std_{key}", state_dict[std_name].to(dtype=dtype))
        return super().load_state_dict(state_dict, strict=False, assign=assign)




