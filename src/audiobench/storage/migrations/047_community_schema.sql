-- Migration 047: Community Detection Schema ("Schools of Thought")
-- Creates expression_communities table and adds community_id to expressions.

CREATE TABLE IF NOT EXISTS expression_communities (
    community_id          INTEGER PRIMARY KEY,
    name                  TEXT,
    name_source           TEXT CHECK(name_source IN ('llm_gemini','llm_ollama','fallback','pending'))
                          NOT NULL DEFAULT 'fallback',
    is_active             INTEGER NOT NULL DEFAULT 1,
    medoid_expr_id        INTEGER REFERENCES expressions(id) ON DELETE SET NULL,
    node_count            INTEGER NOT NULL DEFAULT 0,
    medoid_at_node_count  INTEGER NOT NULL DEFAULT 0,
    resolution            REAL NOT NULL DEFAULT 0.6,
    seed                  INTEGER NOT NULL DEFAULT 42,
    indexed_at            DATETIME,
    updated_at            DATETIME NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%SZ','now'))
);

-- Column on expressions for cluster assignment
ALTER TABLE expressions ADD COLUMN community_id INTEGER REFERENCES expression_communities(community_id);

-- Fast lookup for school-scoped queries (\scope school:N)
CREATE INDEX IF NOT EXISTS idx_expressions_community ON expressions(community_id);
