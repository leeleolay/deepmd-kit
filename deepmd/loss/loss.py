from abc import ABCMeta
from abc import abstractmethod
from typing import Dict
from typing import Tuple

from deepmd.env import tf


class Loss(metaclass=ABCMeta):
    """The abstract class for the loss function."""

    # @abstractmethod
    def build(
        self,
        learning_rate: tf.Tensor,
        natoms: tf.Tensor,
        model_dict: Dict[str, tf.Tensor],
        label_dict: Dict[str, tf.Tensor],
        suffix: str,
    ) -> Tuple[tf.Tensor, Dict[str, tf.Tensor]]:
        """Build the loss function graph.

        Parameters
        ----------
        learning_rate : tf.Tensor
            learning rate
        natoms : tf.Tensor
            number of atoms
        model_dict : dict[str, tf.Tensor]
            A dictionary that maps model keys to tensors
        label_dict : dict[str, tf.Tensor]
            A dictionary that maps label keys to tensors
        suffix : str
            suffix

        Returns
        -------
        tf.Tensor
            the total squared loss
        dict[str, tf.Tensor]
            A dictionary that maps loss keys to more loss tensors
        """
        pass

    @abstractmethod
    def eval(
        self,
        sess: tf.Session,
        feed_dict: Dict[tf.placeholder, tf.Tensor],
        natoms: tf.Tensor,
    ) -> dict:
        """Eval the loss function.

        Parameters
        ----------
        sess : tf.Session
            TensorFlow session
        feed_dict : dict[tf.placeholder, tf.Tensor]
            A dictionary that maps graph elements to values
        natoms : tf.Tensor
            number of atoms

        Returns
        -------
        dict
            A dictionary that maps keys to values. It
            should contain key `natoms`
        """
