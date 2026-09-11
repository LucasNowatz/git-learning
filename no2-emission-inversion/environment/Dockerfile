FROM python:3.13-slim-bookworm

RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        ca-certificates \
        curl \
        git \
        libhdf5-dev \
        libnetcdf-dev \
        nco \
        netcdf-bin \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir \
        numpy==2.2.6 \
        scipy==1.15.3 \
        netCDF4==1.7.2 \
        xarray==2025.10.1 \
        pandas==2.3.3 \
        matplotlib==3.10.6

WORKDIR /app
COPY data /app/data
