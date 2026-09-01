# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import queue
import threading

import pytest

from vllm.v1.ple_offload import connector as connector_module
from vllm.v1.ple_offload.connector import PleOffloadConnector


def _make_connector(
    request_queue: queue.Queue,
) -> PleOffloadConnector:
    """Build the minimal object `_launch` touches, without CUDA or ZMQ."""
    instance = object.__new__(PleOffloadConnector)
    instance.tp_rank = 0
    instance.dp_rank = 0
    instance._uses_cuda_inputs = False
    instance._request_queue = request_queue
    return instance


def test_launch_waits_for_a_slow_staging_thread():
    """Pipeline parallelism can start the next forward before staging drains
    the previous request. The launch must wait instead of failing."""
    request_queue: queue.Queue = queue.Queue(maxsize=1)
    request_queue.put_nowait(object())
    instance = _make_connector(request_queue)

    drained = threading.Event()

    def drain() -> None:
        request_queue.get()
        drained.set()

    consumer = threading.Timer(0.2, drain)
    consumer.start()
    try:
        instance._launch(num_reqs=2, num_tokens=8)
    finally:
        consumer.join()

    assert drained.is_set()
    staged = request_queue.get_nowait()
    assert staged.num_reqs == 2
    assert staged.num_tokens == 8


def test_launch_reports_a_stuck_staging_thread(monkeypatch):
    request_queue: queue.Queue = queue.Queue(maxsize=1)
    request_queue.put_nowait(object())
    instance = _make_connector(request_queue)

    monkeypatch.setattr(connector_module, "PLE_LAUNCH_QUEUE_TIMEOUT_S", 0.05)
    with pytest.raises(RuntimeError, match="did not drain"):
        instance._launch(num_reqs=1, num_tokens=1)


def test_launch_is_a_no_op_off_tp_rank_zero():
    request_queue: queue.Queue = queue.Queue(maxsize=1)
    instance = _make_connector(request_queue)
    instance.tp_rank = 1

    instance._launch(num_reqs=1, num_tokens=1)

    assert request_queue.empty()
