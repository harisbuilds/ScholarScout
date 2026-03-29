-- Schema for ScholarScout universities data

CREATE database scholarscout


CREATE TABLE IF NOT EXISTS universities (
    id              SERIAL PRIMARY KEY,
    program_name    TEXT NOT NULL,
    university_name TEXT NOT NULL,
    field_category  TEXT,
    deadline_ir     DATE,
    deadline_non_ir DATE,
    funding_type    TEXT
);


