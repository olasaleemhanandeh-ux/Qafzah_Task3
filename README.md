# Qafzah Task 3 - MLOps Late Delivery Prediction Pipeline

An end-to-end MLOps pipeline for predicting late delivery risks in e-commerce orders. Built with Scikit-Learn, MLflow, Great Expectations, FastAPI, and Docker.

---

## Project Architecture & Pipeline
1. **Data Validation**: Great Expectations suites for input validation.
2. **Model Tracking & Registry**: MLflow tracking server and model registry (`LateDeliveryModel`).
3. **Automated Testing**: Complete test suite using `pytest` covering preprocessing, prediction, and API routes (13/13 passing).
4. **API & Containerization**: FastAPI app containerized using Docker and Docker Compose.

---

## Quick Start Guide

### 1. Run Unit & Integration Tests
```bash
pytest

### 2. Run with Docker Compose
Bash

docker compose up -d --build

## API Endpoints
🟢 Health Check

    GET http://localhost:8000/health

🔮 Make Predictions

    POST http://localhost:8000/predict

    Request Payload Example:

JSON

{
  "order": {
    "Type": "DEBIT",
    "Days for shipping (real)": 3,
    "Days for shipment (scheduled)": 4,
    "Benefit per order": 29.25,
    "Sales per customer": 300.0,
    "Delivery Status": "Advance shipping",
    "Late_delivery_risk": 0,
    "Category Id": 73,
    "Category Name": "Sporting Goods",
    "Customer City": "Caguas",
    "Customer Country": "Puerto Rico",
    "Customer Segment": "Consumer",
    "Customer Zipcode": 725,
    "customer_zip_code_prefix": 725,
    "customer_state": "PR",
    "Department Id": 2,
    "Department Name": "Fitness",
    "Market": "Pacific Asia",
    "Order City": "Bekasi",
    "Order Country": "Indonesia",
    "Order Item Discount": 0.0,
    "Order Item Discount Rate": 0.0,
    "Order Item Product Price": 327.75,
    "Order Item Profit Ratio": 0.09,
    "Order Item Quantity": 1,
    "Sales": 327.75,
    "Order Item Total": 327.75,
    "Order State": "Jawa Barat",
    "Product Category Id": 73,
    "Product Price": 327.75,
    "Shipping Mode": "Standard Class",
    "order_purchase_timestamp": "2018-01-31 08:00:00",
    "order_approved_at": "2018-01-31 08:30:00",
    "order_delivered_carrier_date": "2018-02-01 10:00:00",
    "order_delivered_customer_date": "2018-02-03 14:00:00",
    "order_estimated_delivery_date": "2018-02-04 00:00:00",
    "shipping_limit_date": "2018-02-02 00:00:00",
    "total_items": 1,
    "total_price": 327.75,
    "total_freight": 15.50,
    "unique_products": 1,
    "unique_sellers": 1,
    "total_payments": 343.25,
    "payment_count": 1,
    "payment_types": 1
  }
}

## Repository Structure

    src/: Data validation, model registration, and FastAPI code.

    tests/: Automated unit and API tests.

    requirements/: Dependencies (runtime.txt & dev.txt).

    Dockerfile & docker-compose.yml: Deployment configurations.
