from typing import List
from typing import Optional

import numpy as np

from deepmd.env import MODEL_VERSION
from deepmd.env import global_cvt_2_ener_float
from deepmd.env import op_module
from deepmd.env import paddle
from deepmd.env import tf
from deepmd.utils.pair_tab import PairTab
from deepmd.utils.spin import Spin

from .model import Model
from .model_stat import make_stat_input
from .model_stat import merge_sys_stat


class EnerModel(Model, paddle.nn.Layer):
    """Energy model.

    Parameters
    ----------
    descriptor
            Descriptor
    fitting_net
            Fitting net
    type_embedding
        Type embedding net
    type_map
            Mapping atom type to the name (str) of the type.
            For example `type_map[1]` gives the name of the type 1.
    data_stat_nbatch
            Number of frames used for data statistic
    data_stat_protect
            Protect parameter for atomic energy regression
    use_srtab
            The table for the short-range pairwise interaction added on top of DP. The table is a text data file with (N_t + 1) * N_t / 2 + 1 columes. The first colume is the distance between atoms. The second to the last columes are energies for pairs of certain types. For example we have two atom types, 0 and 1. The columes from 2nd to 4th are for 0-0, 0-1 and 1-1 correspondingly.
    smin_alpha
            The short-range tabulated interaction will be swithed according to the distance of the nearest neighbor. This distance is calculated by softmin. This parameter is the decaying parameter in the softmin. It is only required when `use_srtab` is provided.
    sw_rmin
            The lower boundary of the interpolation between short-range tabulated interaction and DP. It is only required when `use_srtab` is provided.
    sw_rmin
            The upper boundary of the interpolation between short-range tabulated interaction and DP. It is only required when `use_srtab` is provided.
    srtab_add_bias : bool
        Whether add energy bias from the statistics of the data to short-range tabulated atomic energy. It only takes effect when `use_srtab` is provided.
    spin
        spin
    data_stat_nsample
        The number of training samples in a system to compute and change the energy bias.
    """

    model_type = "ener"

    def __init__(
        self,
        descriptor: dict,
        fitting_net: dict,
        type_embedding: Optional[Union[dict, TypeEmbedNet]] = None,
        type_map: Optional[List[str]] = None,
        data_stat_nbatch: int = 10,
        data_stat_protect: float = 1e-2,
        use_srtab: Optional[str] = None,
        smin_alpha: Optional[float] = None,
        sw_rmin: Optional[float] = None,
        sw_rmax: Optional[float] = None,
        srtab_add_bias: bool = True,
        spin: Optional[Spin] = None,
        data_bias_nsample: int = 10,
        **kwargs,
    ) -> None:
        super().__init__()
        # super(EnerModel, self).__init__(name_scope="EnerModel")
        """Constructor."""
        super().__init__(
            descriptor=descriptor,
            fitting_net=fitting_net,
            type_embedding=type_embedding,
            type_map=type_map,
            data_stat_nbatch=data_stat_nbatch,
            data_bias_nsample=data_bias_nsample,
            data_stat_protect=data_stat_protect,
            use_srtab=use_srtab,
            smin_alpha=smin_alpha,
            sw_rmin=sw_rmin,
            sw_rmax=sw_rmax,
            spin=spin,
            srtab_add_bias=srtab_add_bias,
            **kwargs,
        )
        self.numb_fparam = self.fitting.get_numb_fparam()
        self.numb_aparam = self.fitting.get_numb_aparam()

        # self.type_map = " ".join(self.type_map)
        self.t_tmap = " ".join(self.type_map)
        self.t_mt = self.model_type
        self.t_ver = MODEL_VERSION

    def get_rcut(self):
        return self.rcut

    def get_ntypes(self):
        return self.ntypes

    def get_type_map(self):
        return self.type_map

    def get_numb_fparam(self) -> int:
        """Get the number of frame parameters."""
        return self.numb_fparam

    def get_numb_aparam(self) -> int:
        """Get the number of atomic parameters."""
        return self.numb_aparam

    def data_stat(self, data):
        all_stat = make_stat_input(data, self.data_stat_nbatch, merge_sys=False)
        m_all_stat = merge_sys_stat(all_stat)
        self._compute_input_stat(
            m_all_stat, protection=self.data_stat_protect, mixed_type=data.mixed_type
        )
        self._compute_output_stat(all_stat, mixed_type=data.mixed_type)
        # self.bias_atom_e = data.compute_energy_shift(self.rcond)

    def _compute_input_stat(self, all_stat, protection=1e-2, mixed_type=False):
        if mixed_type:
            self.descrpt.compute_input_stats(
                all_stat["coord"],
                all_stat["box"],
                all_stat["type"],
                all_stat["natoms_vec"],
                all_stat["default_mesh"],
                all_stat,
                mixed_type,
                all_stat["real_natoms_vec"],
            )
        else:
            self.descrpt.compute_input_stats(
                all_stat["coord"],
                all_stat["box"],
                all_stat["type"],
                all_stat["natoms_vec"],
                all_stat["default_mesh"],
                all_stat,
            )
        self.fitting.compute_input_stats(all_stat, protection=protection)

    def _compute_output_stat(self, all_stat, mixed_type=False):
        if mixed_type:
            self.fitting.compute_output_stats(all_stat, mixed_type=mixed_type)
        else:
            self.fitting.compute_output_stats(all_stat)

    def forward(
        self,
        coord_,
        atype_,
        natoms,
        box,
        mesh,
        input_dict,
        frz_model=None,
        ckpt_meta: Optional[str] = None,
        suffix="",
        reuse=None,
    ):
        if input_dict is None:
            input_dict = {}

        coord = paddle.reshape(coord_, [-1, natoms[1] * 3])
        atype = paddle.reshape(atype_, [-1, natoms[1]])
        # input_dict["nframes"] = paddle.shape(coord)[0]  # 推理模型导出的时候注释掉这里，否则会报错

        input_dict["atype"] = atype_

        dout = self.descrpt(
            coord,
            atype,
            natoms,
            box,
            mesh,
            input_dict,
            suffix=suffix,
            reuse=reuse,
        )

        atom_ener = self.fitting(dout, natoms, input_dict, reuse=reuse, suffix=suffix)
        self.atom_ener = atom_ener

        energy_raw = atom_ener

        nloc_atom = (
            natoms[0]
            if self.spin is None
            else paddle.sum(natoms[2 : 2 + len(self.spin.use_spin)]).item()
        )
        energy_raw = paddle.reshape(
            energy_raw, [-1, nloc_atom], name="o_atom_energy" + suffix
        )
        energy = paddle.sum(energy_raw, axis=1, name="o_energy" + suffix)

        force, virial, atom_virial = self.descrpt.prod_force_virial(atom_ener, natoms)

        force = paddle.reshape(force, [-1, 3 * natoms[1]])
        if self.spin is not None:
            # split and concatenate force to compute local atom force and magnetic force
            judge = paddle.equal(natoms[0], natoms[1])
            force = paddle.where(
                judge,
                self.natoms_match(force, natoms),
                self.natoms_not_match(force, natoms, atype),
            )

        force = paddle.reshape(force, [-1, 3 * natoms[1]], name="o_force" + suffix)

        virial = paddle.reshape(virial, [-1, 9], name="o_virial" + suffix)
        atom_virial = paddle.reshape(
            atom_virial, [-1, 9 * natoms[1]], name="o_atom_virial" + suffix
        )

        model_dict = {}
        model_dict["energy"] = energy
        model_dict["force"] = force
        model_dict["virial"] = virial
        model_dict["atom_ener"] = energy_raw
        model_dict["atom_virial"] = atom_virial
        model_dict["coord"] = coord
        model_dict["atype"] = atype

        return model_dict

    def init_variables(
        self,
        graph: tf.Graph,
        graph_def: tf.GraphDef,
        model_type: str = "original_model",
        suffix: str = "",
    ) -> None:
        """Init the embedding net variables with the given frozen model.

        Parameters
        ----------
        graph : tf.Graph
            The input frozen model graph
        graph_def : tf.GraphDef
            The input frozen model graph_def
        model_type : str
            the type of the model
        suffix : str
            suffix to name scope
        """
        # self.frz_model will control the self.model to import the descriptor from the given frozen model instead of building from scratch...
        # initialize fitting net with the given compressed frozen model
        if model_type == "original_model":
            self.descrpt.init_variables(graph, graph_def, suffix=suffix)
            self.fitting.init_variables(graph, graph_def, suffix=suffix)
            tf.constant("original_model", name="model_type", dtype=tf.string)
        elif model_type == "compressed_model":
            self.fitting.init_variables(graph, graph_def, suffix=suffix)
            tf.constant("compressed_model", name="model_type", dtype=tf.string)
        else:
            raise RuntimeError("Unknown model type %s" % model_type)
        if (
            self.typeebd is not None
            and self.typeebd.type_embedding_net_variables is None
        ):
            self.typeebd.init_variables(
                graph, graph_def, suffix=suffix, model_type=model_type
            )

    def natoms_match(self, force, natoms):
        use_spin = self.spin.use_spin
        virtual_len = self.spin.virtual_len
        spin_norm = self.spin.spin_norm
        natoms_index = tf.concat([[0], tf.cumsum(natoms[2:])], axis=0)
        force_real_list = []
        for idx, use in enumerate(use_spin):
            if use is True:
                force_real_list.append(
                    tf.slice(
                        force, [0, natoms_index[idx] * 3], [-1, natoms[idx + 2] * 3]
                    )
                    + tf.slice(
                        force,
                        [0, natoms_index[idx + len(use_spin)] * 3],
                        [-1, natoms[idx + 2 + len(use_spin)] * 3],
                    )
                )
            else:
                force_real_list.append(
                    tf.slice(
                        force, [0, natoms_index[idx] * 3], [-1, natoms[idx + 2] * 3]
                    )
                )
        force_mag_list = []
        for idx, use in enumerate(use_spin):
            if use is True:
                force_mag_list.append(
                    tf.slice(
                        force,
                        [0, natoms_index[idx + len(use_spin)] * 3],
                        [-1, natoms[idx + 2 + len(use_spin)] * 3],
                    )
                )
                force_mag_list[idx] *= virtual_len[idx] / spin_norm[idx]

        force_real = tf.concat(force_real_list, axis=1)
        force_mag = tf.concat(force_mag_list, axis=1)
        loc_force = tf.concat([force_real, force_mag], axis=1)
        force = loc_force
        return force

    def natoms_not_match(self, force, natoms, atype):
        # if ghost atoms exist, compute ghost atom force and magnetic force
        # compute ghost atom force and magnetic force
        use_spin = self.spin.use_spin
        virtual_len = self.spin.virtual_len
        spin_norm = self.spin.spin_norm
        loc_force = self.natoms_match(force, natoms)
        aatype = atype[0, :]
        ghost_atype = aatype[natoms[0] :]
        _, _, ghost_natoms = tf.unique_with_counts(ghost_atype)
        ghost_natoms_index = tf.concat([[0], tf.cumsum(ghost_natoms)], axis=0)
        ghost_natoms_index += natoms[0]

        ghost_force_real_list = []
        for idx, use in enumerate(use_spin):
            if use is True:
                ghost_force_real_list.append(
                    tf.slice(
                        force,
                        [0, ghost_natoms_index[idx] * 3],
                        [-1, ghost_natoms[idx] * 3],
                    )
                    + tf.slice(
                        force,
                        [0, ghost_natoms_index[idx + len(use_spin)] * 3],
                        [-1, ghost_natoms[idx + len(use_spin)] * 3],
                    )
                )
            else:
                ghost_force_real_list.append(
                    tf.slice(
                        force,
                        [0, ghost_natoms_index[idx] * 3],
                        [-1, ghost_natoms[idx] * 3],
                    )
                )
        ghost_force_mag_list = []
        for idx, use in enumerate(use_spin):
            if use is True:
                ghost_force_mag_list.append(
                    tf.slice(
                        force,
                        [0, ghost_natoms_index[idx + len(use_spin)] * 3],
                        [-1, ghost_natoms[idx + len(use_spin)] * 3],
                    )
                )
                ghost_force_mag_list[idx] *= virtual_len[idx] / spin_norm[idx]

        ghost_force_real = tf.concat(ghost_force_real_list, axis=1)
        ghost_force_mag = tf.concat(ghost_force_mag_list, axis=1)
        ghost_force = tf.concat([ghost_force_real, ghost_force_mag], axis=1)
        force = tf.concat([loc_force, ghost_force], axis=1)
        return force

    def change_energy_bias(
        self,
        data: DeepmdDataSystem,
        frozen_model: str,
        origin_type_map: list,
        full_type_map: str,
        bias_shift: str = "delta",
    ) -> None:
        """Change the energy bias according to the input data and the pretrained model.

        Parameters
        ----------
        data : DeepmdDataSystem
            The training data.
        frozen_model : str
            The path file of frozen model.
        origin_type_map : list
            The original type_map in dataset, they are targets to change the energy bias.
        full_type_map : str
            The full type_map in pretrained model
        bias_shift : str
            The mode for changing energy bias : ['delta', 'statistic']
            'delta' : perform predictions on energies of target dataset,
                    and do least sqaure on the errors to obtain the target shift as bias.
            'statistic' : directly use the statistic energy bias in the target dataset.
        """
        self.fitting.change_energy_bias(
            data,
            frozen_model,
            origin_type_map,
            full_type_map,
            bias_shift,
            self.data_bias_nsample,
        )
