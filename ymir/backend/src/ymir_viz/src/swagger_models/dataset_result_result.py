# coding: utf-8

from __future__ import absolute_import
from datetime import date, datetime  # noqa: F401

from typing import List, Dict  # noqa: F401

from src.swagger_models.base_model_ import Model
from src.swagger_models.dataset_result_result_negative_info import DatasetResultResultNegativeInfo  # noqa: F401,E501
from src import util


class DatasetResultResult(Model):
    """NOTE: This class is auto generated by the swagger code generator program.

    Do not edit the class manually.
    """
    def __init__(self, total_images_cnt: int=None, class_ids_count: object=None, class_names_count: object=None, ignored_labels: object=None, negative_info: DatasetResultResultNegativeInfo=None):  # noqa: E501
        """DatasetResultResult - a model defined in Swagger

        :param total_images_cnt: The total_images_cnt of this DatasetResultResult.  # noqa: E501
        :type total_images_cnt: int
        :param class_ids_count: The class_ids_count of this DatasetResultResult.  # noqa: E501
        :type class_ids_count: object
        :param class_names_count: The class_names_count of this DatasetResultResult.  # noqa: E501
        :type class_names_count: object
        :param ignored_labels: The ignored_labels of this DatasetResultResult.  # noqa: E501
        :type ignored_labels: object
        :param negative_info: The negative_info of this DatasetResultResult.  # noqa: E501
        :type negative_info: DatasetResultResultNegativeInfo
        """
        self.swagger_types = {
            'total_images_cnt': int,
            'class_ids_count': object,
            'class_names_count': object,
            'ignored_labels': object,
            'negative_info': DatasetResultResultNegativeInfo
        }

        self.attribute_map = {
            'total_images_cnt': 'total_images_cnt',
            'class_ids_count': 'class_ids_count',
            'class_names_count': 'class_names_count',
            'ignored_labels': 'ignored_labels',
            'negative_info': 'negative_info'
        }
        self._total_images_cnt = total_images_cnt
        self._class_ids_count = class_ids_count
        self._class_names_count = class_names_count
        self._ignored_labels = ignored_labels
        self._negative_info = negative_info

    @classmethod
    def from_dict(cls, dikt) -> 'DatasetResultResult':
        """Returns the dict as a model

        :param dikt: A dict.
        :type: dict
        :return: The DatasetResult_result of this DatasetResultResult.  # noqa: E501
        :rtype: DatasetResultResult
        """
        return util.deserialize_model(dikt, cls)

    @property
    def total_images_cnt(self) -> int:
        """Gets the total_images_cnt of this DatasetResultResult.


        :return: The total_images_cnt of this DatasetResultResult.
        :rtype: int
        """
        return self._total_images_cnt

    @total_images_cnt.setter
    def total_images_cnt(self, total_images_cnt: int):
        """Sets the total_images_cnt of this DatasetResultResult.


        :param total_images_cnt: The total_images_cnt of this DatasetResultResult.
        :type total_images_cnt: int
        """

        self._total_images_cnt = total_images_cnt

    @property
    def class_ids_count(self) -> object:
        """Gets the class_ids_count of this DatasetResultResult.


        :return: The class_ids_count of this DatasetResultResult.
        :rtype: object
        """
        return self._class_ids_count

    @class_ids_count.setter
    def class_ids_count(self, class_ids_count: object):
        """Sets the class_ids_count of this DatasetResultResult.


        :param class_ids_count: The class_ids_count of this DatasetResultResult.
        :type class_ids_count: object
        """

        self._class_ids_count = class_ids_count

    @property
    def class_names_count(self) -> object:
        """Gets the class_names_count of this DatasetResultResult.


        :return: The class_names_count of this DatasetResultResult.
        :rtype: object
        """
        return self._class_names_count

    @class_names_count.setter
    def class_names_count(self, class_names_count: object):
        """Sets the class_names_count of this DatasetResultResult.


        :param class_names_count: The class_names_count of this DatasetResultResult.
        :type class_names_count: object
        """

        self._class_names_count = class_names_count

    @property
    def ignored_labels(self) -> object:
        """Gets the ignored_labels of this DatasetResultResult.


        :return: The ignored_labels of this DatasetResultResult.
        :rtype: object
        """
        return self._ignored_labels

    @ignored_labels.setter
    def ignored_labels(self, ignored_labels: object):
        """Sets the ignored_labels of this DatasetResultResult.


        :param ignored_labels: The ignored_labels of this DatasetResultResult.
        :type ignored_labels: object
        """

        self._ignored_labels = ignored_labels

    @property
    def negative_info(self) -> DatasetResultResultNegativeInfo:
        """Gets the negative_info of this DatasetResultResult.


        :return: The negative_info of this DatasetResultResult.
        :rtype: DatasetResultResultNegativeInfo
        """
        return self._negative_info

    @negative_info.setter
    def negative_info(self, negative_info: DatasetResultResultNegativeInfo):
        """Sets the negative_info of this DatasetResultResult.


        :param negative_info: The negative_info of this DatasetResultResult.
        :type negative_info: DatasetResultResultNegativeInfo
        """

        self._negative_info = negative_info
