# -*- coding:utf-8 -*-
# Author: hankcs
# Date: 2020-10-17 20:30
from abc import ABC
from copy import copy

import hanlp
from hanlp.common.torch_component import TorchComponent
from hanlp.components.distillation.losses import KnowledgeDistillationLoss
from hanlp.components.distillation.schedulers import TemperatureScheduler
from hanlp.utils.torch_util import cuda_devices
from hanlp_common.util import merge_locals_kwargs


class DistillableComponent(TorchComponent, ABC):

    # noinspection PyMethodMayBeStatic,PyTypeChecker
    def build_teacher(self, teacher: str, devices) -> TorchComponent:
        return hanlp.load(teacher, load_kwargs={'devices': devices})

    def distill(self,
                teacher: str,
                trn_data,
                dev_data,
                save_dir,
                batch_size=None,
                epochs=None,
                kd_criterion='kd_ce_loss',
                temperature_scheduler='flsw',
                devices=None,
                logger=None,
                seed=None,
                **kwargs):
        devices = devices or cuda_devices()
        if isinstance(kd_criterion, str):
            kd_criterion = KnowledgeDistillationLoss(kd_criterion)
        if isinstance(temperature_scheduler, str):
            temperature_scheduler = TemperatureScheduler.from_name(temperature_scheduler)
        teacher = self.build_teacher(teacher, devices=devices)
        self.vocabs = teacher.vocabs
        config = copy(teacher.config)
        batch_size = batch_size or config.get('batch_size', None)
        epochs = epochs or config.get('epochs', None)
        config.update(kwargs)
        return super().fit(**merge_locals_kwargs(locals(),
                                                 config,
                                                 excludes=('self', 'kwargs', '__class__', 'config')))

    @property
    def _savable_config(self):
        config = super(DistillableComponent, self)._savable_config
        if 'teacher' in config:
            config.teacher = config.teacher.load_path
        return config
