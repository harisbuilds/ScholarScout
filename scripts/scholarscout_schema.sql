-- Schema for ScholarScout universities data

CREATE database scholarscout


CREATE TABLE universities (
    id              SERIAL PRIMARY KEY,
    program_name    TEXT NOT NULL,
    university_name TEXT NOT NULL,
    field_category  TEXT,
    deadline_ir     DATE,
    deadline_non_ir DATE,
    funding_type    TEXT
);


CREATE TABLE finances (
	university_id int4 NOT NULL,
	tuition_fee_eu_per_year int4 NULL,
	tuition_fee_non_eu_per_year int4 NULL,
	scholarship_available bool NULL,
	funding_category jsonb NULL,
	urls jsonb NULL,
	CONSTRAINT finances_pkey PRIMARY KEY (university_id),
	CONSTRAINT finances_university_id_fkey FOREIGN KEY (university_id) REFERENCES public.universities(id)
);