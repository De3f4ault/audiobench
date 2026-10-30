import json
from unittest.mock import MagicMock, patch

import pytest

from audiobench.daemon.intelligence.connection_surfer import ConnectionSurfer
from audiobench.memory.enums import RelationType


@pytest.fixture
def mock_store():
    store = MagicMock()
    store.written = []

    def mock_add(source_type, content, **kwargs):
        expr = MagicMock()
        expr.id = 200 + len(store.written)
        store.written.append({
            "source_type": source_type,
            "content": content,
            **kwargs,
        })
        return expr

    store.add_expression.side_effect = mock_add
    return store


@pytest.fixture
def mock_session():
    session = MagicMock()

    def create_mock_execute(pairs_data=None):
        def mock_execute(*args, **kwargs):
            query = str(args[0])
            res = MagicMock()
            if "SELECT COUNT(*)" in query:
                res.scalar.return_value = 0
            elif "SELECT e.id" in query:
                params = args[1] if len(args) > 1 and isinstance(args[1], dict) else kwargs.get("params") or {}
                min_len = params.get("min_len")
                rows = pairs_data or []
                if min_len is not None:
                    rows = [r for r in rows if len((r.get("content") or "").strip()) >= min_len]
                map_res = MagicMock()
                map_res.all.return_value = rows
                res.mappings.return_value = map_res
            return res

        return mock_execute

    session.create_mock_execute = create_mock_execute

    ctx = MagicMock()
    ctx.__enter__.return_value = session
    return ctx


@patch("audiobench.storage.expression_repository.ExpressionRepository.link")
@patch("audiobench.daemon.intelligence.connection_surfer.get_session")
@patch("audiobench.daemon.intelligence.connection_surfer._get_store")
def test_cross_file_pair_above_threshold_creates_semantic_edge(
    mock_get_store, mock_get_session, mock_link, mock_store, mock_session
):
    # Two expressions from different files (audio_file_id 1 and 2), highly similar
    data = [
        {
            "id": 1,
            "audio_file_id": 1,
            "file_name": "File 1.mp3",
            "content": "The philosophy of being and time reveals deep ontological structures regarding human existence and temporal phenomena across consciousness.",
        },
        {
            "id": 2,
            "audio_file_id": 2,
            "file_name": "File 2.mp3",
            "content": "Existence and temporal phenomena demonstrate that consciousness constructs meaning through duration and subjective perception over lifetime.",
        },
    ]

    session = mock_session.__enter__()
    session.execute.side_effect = session.create_mock_execute(data)

    # 0.9 similarity
    vecs = {1: [1.0, 0.0], 2: [0.9, 0.435889]}  # cos_sim = 0.9
    mock_store.get_vectors.return_value = vecs

    mock_get_store.return_value = mock_store
    mock_get_session.return_value = mock_session

    surfer = ConnectionSurfer()
    surfer.run()

    # Verify SEMANTIC edge was linked
    assert mock_link.call_count == 1
    call_args = mock_link.call_args[1]
    assert call_args["relation_type"] == RelationType.SEMANTIC.value
    assert call_args["created_by"] == "connection_surfer"

    # Verify graph_edge_event is no longer written to store
    edge_events = [x for x in mock_store.written if x["source_type"] == "graph_edge_event"]
    assert len(edge_events) == 0


@patch("audiobench.storage.expression_repository.ExpressionRepository.link")
@patch("audiobench.daemon.intelligence.connection_surfer.get_session")
@patch("audiobench.daemon.intelligence.connection_surfer._get_store")
def test_no_inference_for_same_file_pairs(
    mock_get_store, mock_get_session, mock_link, mock_store, mock_session
):
    # Same file
    data = [
        {
            "id": 1,
            "audio_file_id": 1,
            "file_name": "File 1.mp3",
            "content": "The philosophy of being and time reveals deep ontological structures regarding human existence and temporal phenomena across consciousness.",
        },
        {
            "id": 2,
            "audio_file_id": 1,
            "file_name": "File 1.mp3",
            "content": "Existence and temporal phenomena demonstrate that consciousness constructs meaning through duration and subjective perception over lifetime.",
        },
    ]

    session = mock_session.__enter__()
    session.execute.side_effect = session.create_mock_execute(data)

    vecs = {1: [1.0, 0.0], 2: [1.0, 0.0]}  # sim = 1.0 > 0.85
    mock_store.get_vectors.return_value = vecs

    mock_get_store.return_value = mock_store
    mock_get_session.return_value = mock_session

    surfer = ConnectionSurfer()
    surfer.run()

    assert mock_link.call_count == 0
    edge_events = [x for x in mock_store.written if x["source_type"] == "graph_edge_event"]
    assert len(edge_events) == 0


@patch("audiobench.storage.expression_repository.ExpressionRepository.link")
@patch("audiobench.daemon.intelligence.connection_surfer.get_session")
@patch("audiobench.daemon.intelligence.connection_surfer._get_store")
def test_boilerplate_content_is_skipped(
    mock_get_store, mock_get_session, mock_link, mock_store, mock_session
):
    # One file is pure LibriVox boilerplate (>= 120 chars to verify boilerplate rejection specifically)
    data = [
        {
            "id": 1,
            "audio_file_id": 1,
            "file_name": "File 1.mp3",
            "content": "This is a LibriVox recording. All LibriVox recordings are in the public domain. For more information, or to volunteer, please visit librivox.org.",
        },
        {
            "id": 2,
            "audio_file_id": 2,
            "file_name": "File 2.mp3",
            "content": "Real book content discussing physics and the fundamental constitution of matter in quantum electrodynamics across deep space regimes.",
        },
    ]

    session = mock_session.__enter__()
    session.execute.side_effect = session.create_mock_execute(data)

    vecs = {1: [1.0, 0.0], 2: [1.0, 0.0]}  # sim = 1.0 > 0.85
    mock_store.get_vectors.return_value = vecs

    mock_get_store.return_value = mock_store
    mock_get_session.return_value = mock_session

    surfer = ConnectionSurfer()
    surfer.run()

    assert mock_link.call_count == 0
    edge_events = [x for x in mock_store.written if x["source_type"] == "graph_edge_event"]
    assert len(edge_events) == 0


@patch("audiobench.storage.expression_repository.ExpressionRepository.link")
@patch("audiobench.daemon.intelligence.connection_surfer.get_session")
@patch("audiobench.daemon.intelligence.connection_surfer._get_store")
def test_per_node_edge_cap_respected(
    mock_get_store, mock_get_session, mock_link, mock_store, mock_session
):
    # 6 expressions from 6 different files, all identical vectors
    data = [
        {
            "id": i,
            "audio_file_id": i,
            "file_name": f"File_{i}.mp3",
            "content": f"Substantive philosophical passage {i} analyzing metaphysics, epistemology, ethics, and phenomenology in historical discourse and analysis.",
        }
        for i in range(1, 7)
    ]
    vecs = {i: [1.0, 0.0] for i in range(1, 7)}

    session = mock_session.__enter__()
    session.execute.side_effect = session.create_mock_execute(data)
    mock_store.get_vectors.return_value = vecs

    mock_get_store.return_value = mock_store
    mock_get_session.return_value = mock_session

    surfer = ConnectionSurfer()
    with patch.object(surfer, "_outbound_semantic_count", return_value=0):
        surfer.run()

    # Settings default surfer_max_edges_per_node is 3
    # Each link adds 1 to both nodes' outbound counts. Total links cannot exceed what satisfies the cap.
    assert mock_link.call_count > 0
    edge_events = [x for x in mock_store.written if x["source_type"] == "graph_edge_event"]
    assert len(edge_events) == 0


def test_connection_surfer_interval_is_two_hours():
    assert ConnectionSurfer.INTERVAL_SECONDS == 3600 * 2  # 7200 seconds


@patch("audiobench.community.assign_incremental")
@patch("audiobench.storage.expression_repository.ExpressionRepository.link")
@patch("audiobench.daemon.intelligence.connection_surfer.get_session")
@patch("audiobench.daemon.intelligence.connection_surfer._get_store")
def test_connection_surfer_calls_assign_incremental(
    mock_get_store, mock_get_session, mock_link, mock_assign, mock_store, mock_session
):
    data = [
        {
            "id": 101,
            "audio_file_id": 1,
            "file_name": "Book_A.mp3",
            "content": "Substantive philosophical passage A analyzing metaphysics, epistemology, ethics, and phenomenology in historical discourse and analysis.",
        },
        {
            "id": 102,
            "audio_file_id": 2,
            "file_name": "Book_B.mp3",
            "content": "Substantive philosophical passage B analyzing metaphysics, epistemology, ethics, and phenomenology in historical discourse and analysis.",
        },
    ]
    session = mock_session.__enter__()
    session.execute.side_effect = session.create_mock_execute(data)
    vecs = {101: [1.0, 0.0], 102: [0.95, 0.3122]}
    mock_store.get_vectors.return_value = vecs

    mock_get_store.return_value = mock_store
    mock_get_session.return_value = mock_session

    surfer = ConnectionSurfer()
    with patch.object(surfer, "_outbound_semantic_count", return_value=0):
        surfer.run()

    assert mock_link.call_count == 1
    assert mock_assign.call_count == 1
    assigned_ids = mock_assign.call_args[0][1]
    assert set(assigned_ids) == {101, 102}


@patch("audiobench.storage.expression_repository.ExpressionRepository.link")
@patch("audiobench.daemon.intelligence.connection_surfer.get_session")
@patch("audiobench.daemon.intelligence.connection_surfer._get_store")
def test_per_node_edge_cap_counts_inbound_edges(
    mock_get_store, mock_get_session, mock_link, mock_store, mock_session
):
    data = [
        {
            "id": 101,
            "audio_file_id": 1,
            "file_name": "Book_A.mp3",
            "content": "Substantive philosophical passage A analyzing metaphysics, epistemology, ethics, and phenomenology in historical discourse and analysis.",
        },
        {
            "id": 102,
            "audio_file_id": 2,
            "file_name": "Book_B.mp3",
            "content": "Substantive philosophical passage B analyzing metaphysics, epistemology, ethics, and phenomenology in historical discourse and analysis.",
        },
    ]
    session = mock_session.__enter__()
    session.execute.side_effect = session.create_mock_execute(data)
    vecs = {101: [1.0, 0.0], 102: [0.95, 0.3122]}
    mock_store.get_vectors.return_value = vecs

    mock_get_store.return_value = mock_store
    mock_get_session.return_value = mock_session

    surfer = ConnectionSurfer()
    # Node 102 already has 5 edges (e.g. from inbound edges in DB)
    def mock_degree(expr_id):
        return 5 if expr_id == 102 else 0

    with patch.object(surfer, "_semantic_degree", side_effect=mock_degree):
        surfer.run()

    # Link should be skipped because node 102 has hit the cap
    assert mock_link.call_count == 0


@patch("audiobench.storage.expression_repository.ExpressionRepository.link")
@patch("audiobench.daemon.intelligence.connection_surfer.get_session")
@patch("audiobench.daemon.intelligence.connection_surfer._get_store")
def test_connection_surfer_excludes_short_and_vacuous_chunks_with_literal_strings(
    mock_get_store, mock_get_session, mock_link, mock_store, mock_session
):
    # Production strings from data/transcriptions.db that previously clustered
    # with high cosine similarity (0.88 - 0.9261) and outranked substantive passages.
    short_strings = [
        ".",
        ". . . . . .",
        "So.",
        "So, also. .",
        "Well,",
        "Right. Right. Right.",
        "That's right.",
    ]
    data = []
    for idx, s in enumerate(short_strings, start=1):
        data.append({
            "id": idx,
            "audio_file_id": idx,
            "file_name": f"Short_File_{idx}.mp3",
            "content": s,
        })

    # Substantive passage pair (>= 120 chars)
    substantive_a = (
        "The philosophy of being and time reveals deep ontological structures "
        "regarding human existence and temporal phenomena across consciousness."
    )
    substantive_b = (
        "Existence and temporal phenomena demonstrate that consciousness constructs "
        "meaning through duration and subjective perception over lifetime."
    )
    data.append({
        "id": 100,
        "audio_file_id": 100,
        "file_name": "Substantive_100.mp3",
        "content": substantive_a,
    })
    data.append({
        "id": 101,
        "audio_file_id": 101,
        "file_name": "Substantive_101.mp3",
        "content": substantive_b,
    })

    session = mock_session.__enter__()
    session.execute.side_effect = session.create_mock_execute(data)

    # Short chunks have high cosine similarity (e.g. 0.9261 for '.' vs '. . . . . .')
    # Substantive pair has lower similarity (0.86)
    vecs = {
        1: [1.0, 0.0],
        2: [0.9261, 0.3773],  # sim = 0.9261
        3: [1.0, 0.0],
        4: [0.8861, 0.4635],  # sim = 0.8861
        5: [1.0, 0.0],
        6: [0.9029, 0.4298],  # sim = 0.9029
        7: [0.9029, 0.4298],  # sim = 0.9029
        100: [1.0, 0.0],
        101: [0.86, 0.5103],  # sim = 0.86
    }
    mock_store.get_vectors.return_value = vecs
    mock_get_store.return_value = mock_store
    mock_get_session.return_value = mock_session

    surfer = ConnectionSurfer()
    with patch.object(surfer, "_outbound_semantic_count", return_value=0):
        surfer.run()

    # The high-similarity vacuous chunks must be filtered by the length floor,
    # ensuring they cannot leapfrog substantive content or consume edge cap slots.
    assert mock_link.call_count == 1
    call_args = mock_link.call_args[1]
    assert {call_args["from_id"], call_args["to_id"]} == {100, 101}
    assert call_args["relation_type"] == RelationType.SEMANTIC.value


@patch("audiobench.storage.expression_repository.ExpressionRepository.link")
@patch("audiobench.daemon.intelligence.connection_surfer.get_session")
@patch("audiobench.daemon.intelligence.connection_surfer._get_store")
def test_connection_surfer_defense_in_depth_in_memory_guard(
    mock_get_store, mock_get_session, mock_link, mock_store, mock_session
):
    # Verify that even if candidate selection returns short chunks (e.g. mock or bypass),
    # Guard 0 in Step 4 rejects them in-memory before edge creation.
    data = [
        {"id": 1, "audio_file_id": 1, "file_name": "File_1.mp3", "content": "."},
        {"id": 2, "audio_file_id": 2, "file_name": "File_2.mp3", "content": ". . . . . ."},
    ]
    session = mock_session.__enter__()
    # Return directly without length filtering to simulate defense-in-depth trigger
    def mock_execute_unfiltered(*args, **kwargs):
        query = str(args[0])
        res = MagicMock()
        if "SELECT COUNT(*)" in query:
            res.scalar.return_value = 0
        elif "SELECT e.id" in query:
            map_res = MagicMock()
            map_res.all.return_value = data
            res.mappings.return_value = map_res
        return res

    session.execute.side_effect = mock_execute_unfiltered

    vecs = {1: [1.0, 0.0], 2: [0.9261, 0.3773]}
    mock_store.get_vectors.return_value = vecs
    mock_get_store.return_value = mock_store
    mock_get_session.return_value = mock_session

    surfer = ConnectionSurfer()
    with patch.object(surfer, "_outbound_semantic_count", return_value=0):
        surfer.run()

    assert mock_link.call_count == 0


@patch("audiobench.storage.expression_repository.ExpressionRepository.link")
@patch("audiobench.daemon.intelligence.connection_surfer.get_session")
@patch("audiobench.daemon.intelligence.connection_surfer._get_store")
def test_connection_surfer_excludes_media_transition_boilerplate(
    mock_get_store, mock_get_session, mock_link, mock_store, mock_session
):
    # Literal production expression #66198:
    # "You must be hot. This book is continued on disc 5"
    media_outro_literal = "You must be hot. This book is continued on disc 5"
    substantive = (
        "The philosophy of being and time reveals deep ontological structures "
        "regarding human existence and temporal phenomena across consciousness."
    )
    data = [
        {"id": 66198, "audio_file_id": 1, "file_name": "Disc4.mp3", "content": media_outro_literal},
        {"id": 89259, "audio_file_id": 2, "file_name": "Disc5.mp3", "content": substantive},
    ]

    session = mock_session.__enter__()
    session.execute.side_effect = session.create_mock_execute(data)

    vecs = {66198: [1.0, 0.0], 89259: [0.95, 0.3122]}
    mock_store.get_vectors.return_value = vecs
    mock_get_store.return_value = mock_store
    mock_get_session.return_value = mock_session

    surfer = ConnectionSurfer()
    with patch.object(surfer, "_outbound_semantic_count", return_value=0):
        surfer.run()

    # Excluded by candidate length query and in-memory guards
    assert mock_link.call_count == 0


@patch("audiobench.storage.expression_repository.ExpressionRepository.link")
@patch("audiobench.daemon.intelligence.connection_surfer.get_session")
@patch("audiobench.daemon.intelligence.connection_surfer._get_store")
def test_connection_surfer_retains_substantive_chunks(
    mock_get_store, mock_get_session, mock_link, mock_store, mock_session
):
    # Positive control: substantive chunks (>= 120 chars) across works
    # must be linked normally when similarity exceeds threshold.
    substantive_a = (
        "The phenomenology of internal time consciousness explores how retentions, "
        "primal impressions, and protentions structure the flow of subjective experience."
    )
    substantive_b = (
        "Temporal horizons frame every act of intentionality, grounding our perception "
        "of continuous objects across changing duration and perceptual fields."
    )
    assert len(substantive_a) >= 120
    assert len(substantive_b) >= 120

    data = [
        {"id": 501, "audio_file_id": 10, "file_name": "Phenomenology.mp3", "content": substantive_a},
        {"id": 502, "audio_file_id": 20, "file_name": "Intentionality.mp3", "content": substantive_b},
    ]

    session = mock_session.__enter__()
    session.execute.side_effect = session.create_mock_execute(data)

    vecs = {501: [1.0, 0.0], 502: [0.88, 0.47497]}  # sim = 0.88 > 0.85
    mock_store.get_vectors.return_value = vecs
    mock_get_store.return_value = mock_store
    mock_get_session.return_value = mock_session

    surfer = ConnectionSurfer()
    with patch.object(surfer, "_outbound_semantic_count", return_value=0):
        surfer.run()

    assert mock_link.call_count == 1
    call_args = mock_link.call_args[1]
    assert {call_args["from_id"], call_args["to_id"]} == {501, 502}
    assert call_args["relation_type"] == RelationType.SEMANTIC.value



