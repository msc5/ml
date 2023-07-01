CREATE DATABASE meta;

USE meta;

CREATE TABLE runs (
    id varchar(255),
    start_time varchar(255),
    version varchar(255),
    name varchar(255)
);

CREATE TABLE metrics (
    name varchar(255),
    start_time varchar(255),
    count varchar(255)
);
