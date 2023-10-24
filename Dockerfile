FROM gurobi/python:10.0.3

# Set the application directory
WORKDIR /OptiRun

# Install system dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends libgdal-dev && \
    rm -rf /var/lib/apt/lists/*

# Copy the application code
COPY . .

# Install Python dependencies
RUN python -m venv venv && \
    . venv/bin/activate && \
    pip install --no-cache-dir -r requirements.txt

# Set environment variables
ENV VIRTUAL_ENV=/OptiRun/venv
ENV GRB_LICENSE_FILE=/OptiRun/gurobi.lic
ENV PATH="${VIRTUAL_ENV}/bin:$PATH"

# Execute the code
CMD ["python", "-u", "run.py"]