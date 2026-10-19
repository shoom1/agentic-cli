-- A session database written by google-adk 1.36.0.
-- Made by tests/fixtures/make_adk1_sessions.py; do not edit by hand.
BEGIN TRANSACTION;
CREATE TABLE adk_internal_metadata (
	"key" VARCHAR(128) NOT NULL,
	value VARCHAR(256) NOT NULL,
	PRIMARY KEY ("key")
);
INSERT INTO "adk_internal_metadata" VALUES('schema_version','1');
CREATE TABLE app_states (
	app_name VARCHAR(128) NOT NULL,
	state TEXT NOT NULL,
	update_time DATETIME NOT NULL,
	PRIMARY KEY (app_name)
);
INSERT INTO "app_states" VALUES('research_demo','{}','2026-10-19 00:09:09');
CREATE TABLE events (
	id VARCHAR(128) NOT NULL,
	app_name VARCHAR(128) NOT NULL,
	user_id VARCHAR(128) NOT NULL,
	session_id VARCHAR(128) NOT NULL,
	invocation_id VARCHAR(256) NOT NULL,
	timestamp DATETIME NOT NULL,
	event_data TEXT,
	PRIMARY KEY (id, app_name, user_id, session_id),
	FOREIGN KEY(app_name, user_id, session_id) REFERENCES sessions (app_name, user_id, id) ON DELETE CASCADE
);
INSERT INTO "events" VALUES('8f10db2e-5d7a-4161-95d9-4983043523b7','research_demo','tester','adk1-session','inv-1','2026-10-18 20:09:09.386772','{"content": {"parts": [{"text": "Search arXiv for speculative decoding."}], "role": "user"}, "invocation_id": "inv-1", "author": "user", "actions": {"state_delta": {}, "artifact_delta": {}, "requested_auth_configs": {}, "requested_tool_confirmations": {}}, "id": "8f10db2e-5d7a-4161-95d9-4983043523b7", "timestamp": 1792368549.386772}');
INSERT INTO "events" VALUES('69e58df7-f202-465e-8df0-ff0175def1b0','research_demo','tester','adk1-session','inv-1','2026-10-18 20:09:09.386813','{"content": {"parts": [{"function_call": {"id": "call-1", "args": {"query": "speculative decoding"}, "name": "search_arxiv"}}], "role": "model"}, "invocation_id": "inv-1", "author": "coordinator", "actions": {"state_delta": {}, "artifact_delta": {}, "requested_auth_configs": {}, "requested_tool_confirmations": {}}, "id": "69e58df7-f202-465e-8df0-ff0175def1b0", "timestamp": 1792368549.386813}');
INSERT INTO "events" VALUES('4e34685c-1c83-4854-9b95-048f73832c6c','research_demo','tester','adk1-session','inv-1','2026-10-18 20:09:09.386834','{"content": {"parts": [{"function_response": {"id": "call-1", "name": "search_arxiv", "response": {"success": true, "count": 2}}}], "role": "user"}, "invocation_id": "inv-1", "author": "coordinator", "actions": {"state_delta": {}, "artifact_delta": {}, "requested_auth_configs": {}, "requested_tool_confirmations": {}}, "id": "4e34685c-1c83-4854-9b95-048f73832c6c", "timestamp": 1792368549.386834}');
INSERT INTO "events" VALUES('4142c4fe-2dff-4cb2-a5ec-de3aa7981ac7','research_demo','tester','adk1-session','inv-1','2026-10-18 20:09:09.386851','{"content": {"parts": [{"function_call": {"id": "call-2", "args": {"code": "1 + 1"}, "name": "sandbox_execute"}}], "role": "model"}, "invocation_id": "inv-1", "author": "coordinator", "actions": {"state_delta": {}, "artifact_delta": {}, "requested_auth_configs": {}, "requested_tool_confirmations": {}}, "long_running_tool_ids": ["call-2"], "id": "4142c4fe-2dff-4cb2-a5ec-de3aa7981ac7", "timestamp": 1792368549.386851}');
INSERT INTO "events" VALUES('72c0f0f1-86ef-45fb-846a-7790a559e031','research_demo','tester','adk1-session','inv-1','2026-10-18 20:09:09.386863','{"content": {"parts": [{"text": "Found 2 papers; the analysis is running."}], "role": "model"}, "invocation_id": "inv-1", "author": "coordinator", "actions": {"state_delta": {}, "artifact_delta": {}, "requested_auth_configs": {}, "requested_tool_confirmations": {}}, "id": "72c0f0f1-86ef-45fb-846a-7790a559e031", "timestamp": 1792368549.386863}');
CREATE TABLE sessions (
	app_name VARCHAR(128) NOT NULL,
	user_id VARCHAR(128) NOT NULL,
	id VARCHAR(128) NOT NULL,
	state TEXT NOT NULL,
	create_time DATETIME NOT NULL,
	update_time DATETIME NOT NULL,
	PRIMARY KEY (app_name, user_id, id)
);
INSERT INTO "sessions" VALUES('research_demo','tester','adk1-session','{"plan": "1. Search arXiv"}','2026-10-19 00:09:09.385366','2026-10-19 00:09:09.386863');
CREATE TABLE user_states (
	app_name VARCHAR(128) NOT NULL,
	user_id VARCHAR(128) NOT NULL,
	state TEXT NOT NULL,
	update_time DATETIME NOT NULL,
	PRIMARY KEY (app_name, user_id)
);
INSERT INTO "user_states" VALUES('research_demo','tester','{"lang": "en"}','2026-10-19 00:09:09');
CREATE INDEX idx_events_app_user_session_ts ON events (app_name, user_id, session_id, timestamp DESC);
COMMIT;
