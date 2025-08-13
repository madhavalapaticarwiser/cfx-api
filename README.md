# cfx-api
FastAPI + Docker container for all CFX price estimates (Retail, Private, Trade-in)

# 1. (Optional)
git clone https://github.com/madhavalapaticarwiser/cfx-api.git
cd cfx-api

# 2. Pull the pre-built image
docker pull madhavalapati/cfx-api:1.0.0

# 3. Run the container
docker run --rm -p 8080:8080 madhavalapati/cfx-api:1.0.0

# 1.1. Build the Docker image (Optional)
docker build -t cfx-api .

# 1.2. Run the container (Optional)
docker run --rm -p 8080:8080 cfx-api

# 4. Verify in browser or via cURL
#    Browser: http://localhost:8080/docs
#    cURL:
curl -X 'POST' \
  'http://localhost:8080/predict' \
  -H 'accept: application/json' \
  -H 'Content-Type: application/json' \
  -d '{
  "year": 2019,
  "make": "Honda",
  "model": "Civic",
  "trim": "EX CVT",
  "mileage": 67000,
  "interior": "great",
  "exterior": "great",
  "mechanical": "great",
  "line": "Economy",
  "drivetrain": "FWD",
  "transmission": "CVT"
}
'

Should return
{
  "success": true,
  "predictions": {
    "Retail": 18409.63671875,
    "Private": 16384.494140625,
    "Trade-In": 13714.265625
  },
  "matched_vehicle": {
    "make": "Honda",
    "model": "Civic",
    "trim": "EX CVT"
  },
  "message": null
}
