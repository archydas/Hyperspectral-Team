-- DS3 PostgreSQL schema (generated from src/ds3/db/models.py; do not hand-edit)

CREATE TABLE materials (
	id SERIAL NOT NULL, 
	name VARCHAR(300) NOT NULL, 
	category VARCHAR(100), 
	PRIMARY KEY (id), 
	UNIQUE (name)
);

CREATE INDEX ix_materials_category ON materials (category);
CREATE TABLE sources (
	id SERIAL NOT NULL, 
	name VARCHAR(200) NOT NULL, 
	kind VARCHAR(30) NOT NULL, 
	url VARCHAR(500), 
	license VARCHAR(200), 
	notes TEXT, 
	PRIMARY KEY (id), 
	UNIQUE (name)
);

CREATE TABLE cubes (
	id SERIAL NOT NULL, 
	source_id INTEGER, 
	name VARCHAR(300) NOT NULL, 
	path VARCHAR(1000) NOT NULL, 
	sha256 VARCHAR(64) NOT NULL, 
	format VARCHAR(30) NOT NULL, 
	rows INTEGER NOT NULL, 
	cols INTEGER NOT NULL, 
	bands INTEGER NOT NULL, 
	unit VARCHAR(20) NOT NULL, 
	sensor VARCHAR(100), 
	processing_level VARCHAR(50), 
	crs VARCHAR(100), 
	transform JSON, 
	wl_min_nm FLOAT NOT NULL, 
	wl_max_nm FLOAT NOT NULL, 
	wavelengths_nm FLOAT[] NOT NULL, 
	fwhm_nm FLOAT[], 
	bad_band_mask BOOLEAN[] NOT NULL, 
	qc JSONB, 
	meta JSONB, 
	contract_version VARCHAR(10) NOT NULL, 
	ingested_at TIMESTAMP WITH TIME ZONE NOT NULL, 
	PRIMARY KEY (id), 
	FOREIGN KEY(source_id) REFERENCES sources (id), 
	UNIQUE (sha256)
);

CREATE TABLE libraries (
	id SERIAL NOT NULL, 
	name VARCHAR(200) NOT NULL, 
	source_id INTEGER, 
	description TEXT, 
	PRIMARY KEY (id), 
	UNIQUE (name), 
	FOREIGN KEY(source_id) REFERENCES sources (id)
);

CREATE TABLE datasets (
	id SERIAL NOT NULL, 
	library_id INTEGER NOT NULL, 
	name VARCHAR(300) NOT NULL, 
	sha256 VARCHAR(64) NOT NULL, 
	format VARCHAR(30) NOT NULL, 
	ingested_at TIMESTAMP WITH TIME ZONE NOT NULL, 
	PRIMARY KEY (id), 
	FOREIGN KEY(library_id) REFERENCES libraries (id), 
	UNIQUE (sha256)
);

CREATE INDEX ix_datasets_library_id ON datasets (library_id);
CREATE TABLE samples (
	id SERIAL NOT NULL, 
	dataset_id INTEGER NOT NULL, 
	material_id INTEGER NOT NULL, 
	label VARCHAR(300) NOT NULL, 
	meta JSONB, 
	PRIMARY KEY (id), 
	FOREIGN KEY(dataset_id) REFERENCES datasets (id), 
	FOREIGN KEY(material_id) REFERENCES materials (id)
);

CREATE INDEX ix_samples_dataset_id ON samples (dataset_id);
CREATE INDEX ix_samples_material_id ON samples (material_id);
CREATE TABLE spectra (
	id SERIAL NOT NULL, 
	sample_id INTEGER NOT NULL, 
	unit VARCHAR(20) NOT NULL, 
	purity VARCHAR(20) NOT NULL, 
	wl_min_nm FLOAT NOT NULL, 
	wl_max_nm FLOAT NOT NULL, 
	n_bands INTEGER NOT NULL, 
	wavelengths_nm FLOAT[] NOT NULL, 
	values FLOAT[] NOT NULL, 
	bad_band_mask BOOLEAN[], 
	meta JSONB, 
	PRIMARY KEY (id), 
	FOREIGN KEY(sample_id) REFERENCES samples (id)
);

CREATE INDEX ix_spectra_sample_id ON spectra (sample_id);
CREATE INDEX ix_spectra_wl ON spectra (wl_min_nm, wl_max_nm);
CREATE INDEX ix_spectra_unit_purity ON spectra (unit, purity);