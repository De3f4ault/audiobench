from unittest.mock import MagicMock, patch

from audiobench.daemon.client import DaemonClient
from audiobench.daemon.factory import get_daemon_client
from audiobench.daemon.local_client import LocalRetrievalClient


@patch("audiobench.daemon.factory._is_socket_alive")
@patch("audiobench.daemon.factory.Path.exists")
def test_factory_returns_daemon_if_alive(mock_exists: MagicMock, mock_is_alive: MagicMock) -> None:
    """If the socket exists and is alive, return DaemonClient immediately."""
    mock_exists.return_value = True
    mock_is_alive.return_value = True

    with patch("audiobench.daemon.factory.DaemonClient") as mock_client_cls:
        mock_client = MagicMock(spec=DaemonClient)
        mock_client_cls.return_value = mock_client

        client = get_daemon_client()
        assert client is mock_client


@patch("audiobench.daemon.factory._is_socket_alive", return_value=False)
@patch("audiobench.daemon.factory._FAST_PATH_TIMEOUT", new=0)  # skip polling loop
def test_factory_fallback_on_start_failure(mock_alive: MagicMock) -> None:
    """If the socket never comes up, get_daemon_client falls back to LocalRetrievalClient."""
    with patch("audiobench.daemon.factory.LocalRetrievalClient") as mock_local_cls:
        mock_local = MagicMock(spec=LocalRetrievalClient)
        mock_local_cls.return_value = mock_local

        client = get_daemon_client()
        assert client is mock_local


@patch("audiobench.daemon.factory._is_socket_alive", return_value=False)
@patch("audiobench.daemon.factory._fork_daemon", return_value=True)
@patch("audiobench.daemon.factory.DaemonClient")
def test_factory_autostart_attempts_fork(
    mock_daemon_cls: MagicMock, mock_fork: MagicMock, mock_alive: MagicMock
) -> None:
    """When socket is dead and autostart=True, get_daemon_client forks the daemon."""
    import audiobench.daemon.factory as factory

    factory._daemon_forked_at = None

    mock_client = MagicMock(spec=DaemonClient)
    mock_client.ping.return_value = True
    mock_daemon_cls.return_value = mock_client

    with patch.object(factory, "_FAST_PATH_TIMEOUT", 10.0):
        client = get_daemon_client(autostart=True)
        mock_fork.assert_called_once()
        assert client is mock_client


@patch("audiobench.daemon.factory._is_socket_alive", return_value=False)
@patch("audiobench.daemon.factory._fork_daemon")
def test_factory_autostart_disabled(mock_fork: MagicMock, mock_alive: MagicMock) -> None:
    """When autostart=False, get_daemon_client does not fork the daemon."""
    import audiobench.daemon.factory as factory

    factory._daemon_forked_at = None

    with patch.object(factory, "_DAEMON_QUICK_WAIT", 0.0), \
         patch("audiobench.daemon.factory.LocalRetrievalClient") as mock_local_cls:
        mock_local = MagicMock(spec=LocalRetrievalClient)
        mock_local_cls.return_value = mock_local

        client = get_daemon_client(autostart=False)
        mock_fork.assert_not_called()
        assert client is mock_local


def test_local_retrieval_client_rerank() -> None:
    """LocalRetrievalClient.rerank predicts scores via get_reranker."""
    from audiobench.daemon.local_client import LocalRetrievalClient

    client = LocalRetrievalClient()
    mock_reranker = MagicMock()
    mock_reranker.predict.return_value = MagicMock(tolist=lambda: [0.95, 0.42])

    with patch("audiobench.memory.singletons.get_reranker", return_value=mock_reranker):
        scores = client.rerank("query text", ["doc1", "doc2"])
        assert scores == [0.95, 0.42]
        mock_reranker.predict.assert_called_once_with([["query text", "doc1"], ["query text", "doc2"]])


def test_local_retrieval_client_does_not_prewarm_on_get_store() -> None:
    """LocalRetrievalClient._get_store does not eagerly load all 4 models."""
    from audiobench.daemon.local_client import LocalRetrievalClient

    client = LocalRetrievalClient()
    with patch("audiobench.memory.memory_store.MemoryStore"), \
         patch("audiobench.memory.singletons.pre_warm_retrieval_pipeline") as mock_prewarm:
        client._get_store()
        mock_prewarm.assert_not_called()

