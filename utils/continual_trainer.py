
import tqdm
import logging
import time
import numpy as np

from collections import deque
from collections.abc import Callable
from dataclasses import asdict

from tianshou.data import (
    CollectStats,
    CollectStatsBase,
    SequenceSummaryStats,
    EpochStats,
    InfoStats,
    TimingStats
)
from tianshou.policy.base import TrainingStats
from tianshou.data.collector import (
    BaseCollector, 
    AsyncCollector,
)
from tianshou.trainer import BaseTrainer
from tianshou.utils import (
    BaseLogger,
    DummyTqdm,
    tqdm_config,
)
from tianshou.utils.logging import set_numerical_fields_to_precision


log = logging.getLogger(__name__)

def test_continual_episode(
    collectors: list[BaseCollector],
    test_fn: Callable[[int, int | None], None] | None,
    epoch: int,
    n_episode: int,
    logger: BaseLogger | None = None,
    global_step: int | None = None,
    reward_metric: Callable[[np.ndarray], np.ndarray] | None = None,
) -> CollectStats:
    """A simple wrapper of testing policy in collector."""
    for i, collector in enumerate(collectors):
        collector.reset(reset_stats=False)
        ## TODO: need to ensure this test-fn doesn't do funny things inside the loop
        if test_fn:
            test_fn(epoch, global_step)
        result = collector.collect(n_episode=n_episode)
        if reward_metric:  # TODO: move into collector
            rew = reward_metric(result.returns)
            result.returns = rew
            result.returns_stat = SequenceSummaryStats.from_sequence(rew)
        if logger and global_step is not None:
            assert result.n_collected_episodes > 0
            logger.log_test_data(asdict(result), global_step, task_id=i)
    return result

def gather_continual_info(
    start_time: float,
    policy_update_time: float,
    gradient_step: int,
    best_reward: float,
    best_reward_std: float,
    train_collector: BaseCollector | None = None,
    test_collector: list[BaseCollector] | None = None,
) -> InfoStats:
    """A simple wrapper of gathering information from collectors.

    :return: InfoStats object with times computed based on the `start_time` and
        episode/step counts read off the collectors. No computation of
        expensive statistics is done here.
    """
    duration = max(0.0, time.time() - start_time)
    test_time = 0.0
    update_speed = 0.0
    train_time_collect = 0.0
    if test_collector is not None:
        test_time = 0 #test_collector.collect_time

    if train_collector is not None:
        train_time_collect = train_collector.collect_time
        update_speed = train_collector.collect_step / (duration - test_time)

    timing_stat = TimingStats(
        total_time=duration,
        train_time=duration - test_time,
        train_time_collect=train_time_collect,
        train_time_update=policy_update_time,
        test_time=test_time,
        update_speed=update_speed,
    )

    return InfoStats(
        gradient_step=gradient_step,
        best_reward=best_reward,
        best_reward_std=best_reward_std,
        train_step=train_collector.collect_step if train_collector is not None else 0,
        train_episode=train_collector.collect_episode if train_collector is not None else 0,
        ## don't care
        test_step=0,
        test_episode=0,
        timing=timing_stat,
    )

class ContinualOffpolicyTrainer(BaseTrainer):

    ## NOTE: from OffpolicyTrainer
    def policy_update_fn(
        self,
        collect_stats: CollectStatsBase,
    ) -> TrainingStats:
        """Perform `update_per_step * n_collected_steps` gradient steps by sampling mini-batches from the buffer.

        :param collect_stats: the :class:`~TrainingStats` instance returned by the last gradient step. Some values
            in it will be replaced by their moving averages.
        """
        assert self.train_collector is not None
        n_collected_steps = collect_stats.n_collected_steps
        n_gradient_steps = round(self.update_per_step * n_collected_steps)
        if n_gradient_steps == 0:
            raise ValueError(
                f"n_gradient_steps is 0, n_collected_steps={n_collected_steps}, "
                f"update_per_step={self.update_per_step}",
            )
        for _ in range(n_gradient_steps):
            update_stat = self._sample_and_update(self.train_collector.buffer)

            # logging
            self.policy_update_time += update_stat.train_time
        # TODO: only the last update_stat is returned, should be improved
        return update_stat
    
    def __next__(self) -> EpochStats:
        """Perform one epoch (both train and eval)."""
        self.epoch += 1
        self.iter_num += 1

        if self.iter_num > 1:
            # iterator exhaustion check
            if self.epoch > self.max_epoch:
                raise StopIteration

            # exit flag 1, when stop_fn succeeds in train_step or test_step
            if self.stop_fn_flag:
                raise StopIteration

        progress = tqdm.tqdm if self.show_progress else DummyTqdm

        # perform n step_per_epoch
        with progress(total=self.step_per_epoch, desc=f"Epoch #{self.epoch}", **tqdm_config) as t:
            train_stat: CollectStatsBase
            while t.n < t.total and not self.stop_fn_flag:
                train_stat, update_stat, self.stop_fn_flag = self.training_step()

                if isinstance(train_stat, CollectStats):
                    pbar_data_dict = {
                        "env_step": str(self.env_step),
                        "rew": f"{self.last_rew:.2f}",
                        "len": str(int(self.last_len)),
                        "n/ep": str(train_stat.n_collected_episodes),
                        "n/st": str(train_stat.n_collected_steps),
                    }
                    t.update(train_stat.n_collected_steps)
                else:
                    pbar_data_dict = {}
                    t.update()

                pbar_data_dict = set_numerical_fields_to_precision(pbar_data_dict)
                pbar_data_dict["gradient_step"] = str(self._gradient_step)
                t.set_postfix(**pbar_data_dict)

                if self.stop_fn_flag:
                    break

            if t.n <= t.total and not self.stop_fn_flag:
                t.update()

        test_stat = None
        if not self.stop_fn_flag:
            self.logger.save_data(
                self.epoch,
                self.env_step,
                self._gradient_step,
                self.save_checkpoint_fn,
            )
            # test
            if self.test_collector is not None:
                test_stat, self.stop_fn_flag = self.test_step()

        info_stat = gather_continual_info(
            start_time=self.start_time,
            policy_update_time=self.policy_update_time,
            gradient_step=self._gradient_step,
            best_reward=self.best_reward,
            best_reward_std=self.best_reward_std,
            train_collector=self.train_collector,
            test_collector=self.test_collector,
        )

        self.logger.log_info_data(asdict(info_stat), self.epoch)

        # in case trainer is used with run(), epoch_stat will not be returned
        return EpochStats(
            epoch=self.epoch,
            train_collect_stat=train_stat,
            test_collect_stat=test_stat,
            training_stat=update_stat,
            info_stat=info_stat,
        )
    
    def test_step(self) -> tuple[CollectStats, bool]:
        """Perform one testing step."""
        assert self.episode_per_test is not None
        assert self.test_collector is not None
        stop_fn_flag = False
        test_stat = test_continual_episode(
            self.test_collector,
            self.test_fn,
            self.epoch,
            self.episode_per_test,
            self.logger,
            self.env_step,
            self.reward_metric,
        )
        assert test_stat.returns_stat is not None  # for mypy
        rew, rew_std = test_stat.returns_stat.mean, test_stat.returns_stat.std
        if self.best_epoch < 0 or self.best_reward < rew:
            self.best_epoch = self.epoch
            self.best_reward = float(rew)
            self.best_reward_std = rew_std
            if self.save_best_fn:
                self.save_best_fn(self.policy)
        log_msg = (
            f"Epoch #{self.epoch}: test_reward: {rew:.6f} ± {rew_std:.6f},"
            f" best_reward: {self.best_reward:.6f} ± "
            f"{self.best_reward_std:.6f} in #{self.best_epoch}"
        )
        log.info(log_msg)
        if self.verbose:
            print(log_msg, flush=True)

        if self.stop_fn and self.stop_fn(self.best_reward):
            stop_fn_flag = True

        return test_stat, stop_fn_flag
        
    ## handle list reset
    def _reset_collectors(self, reset_buffer: bool = False) -> None:
        if self.train_collector is not None:
            self.train_collector.reset(reset_buffer=reset_buffer)
        if self.test_collector is not None:
            for collector in self.test_collector:
                collector.reset(reset_buffer=reset_buffer)

    def reset(self, reset_collectors: bool = True, reset_buffer: bool = False) -> None:
        """Initialize or reset the instance to yield a new iterator from zero."""
        self.is_run = False
        self.env_step = 0
        if self.resume_from_log:
            (
                self.start_epoch,
                self.env_step,
                self._gradient_step,
            ) = self.logger.restore_data()

        self.last_rew, self.last_len = 0.0, 0.0
        self.start_time = time.time()

        if reset_collectors:
            self._reset_collectors(reset_buffer=reset_buffer)

        if self.train_collector is not None and (
            self.train_collector.policy != self.policy or self.test_collector is None
        ):
            self.test_in_train = False

        if self.test_collector is not None:
            assert self.episode_per_test is not None
            assert not isinstance(self.test_collector, AsyncCollector)  # Issue 700
            test_result = test_continual_episode(
                self.test_collector,
                self.test_fn,
                self.start_epoch,
                self.episode_per_test,
                self.logger,
                self.env_step,
                self.reward_metric,
            )
            assert test_result.returns_stat is not None  # for mypy
            self.best_epoch = self.start_epoch
            self.best_reward, self.best_reward_std = (
                test_result.returns_stat.mean,
                test_result.returns_stat.std,
            )
        if self.save_best_fn:
            self.save_best_fn(self.policy)

        self.epoch = self.start_epoch
        self.stop_fn_flag = False
        self.iter_num = 0

    def run(self, reset_prior_to_run: bool = True) -> InfoStats:
        """Consume iterator.

        See itertools - recipes. Use functions that consume iterators at C speed
        (feed the entire iterator into a zero-length deque).
        """
        if reset_prior_to_run:
            self.reset()
        try:
            self.is_run = True
            deque(self, maxlen=0)  # feed the entire iterator into a zero-length deque
            info = gather_continual_info(
                start_time=self.start_time,
                policy_update_time=self.policy_update_time,
                gradient_step=self._gradient_step,
                best_reward=self.best_reward,
                best_reward_std=self.best_reward_std,
                train_collector=self.train_collector,
                test_collector=self.test_collector,
            )
        finally:
            self.is_run = False

        return info
