# AWS Deployment Guide for Water Potability App

This guide outlines the steps to deploy your Dockerized application to AWS using **AWS App Runner**. App Runner is the simplest way to deploy containerized web applications.

## Prerequisites

1.  **AWS Account**: You need an active AWS account.
2.  **AWS CLI**: Installed and configured (`aws configure`).
3.  **Docker**: Running locally.

## Step 1: Create an ECR Repository

Amazon Elastic Container Registry (ECR) is where we will store our Docker image.

1.  Log in to the AWS Console and search for **ECR**.
2.  Click **Create repository**.
3.  Name it `water-potability-app`.
4.  Keep other settings default and click **Create repository**.

## Step 2: Push Docker Image to ECR

1.  **Login to ECR** (Replace `region` and `account_id`):
    ```powershell
    aws ecr get-login-password --region us-east-1 | docker login --username AWS --password-stdin <account_id>.dkr.ecr.us-east-1.amazonaws.com
    ```

2.  **Tag your local image**:
    ```powershell
    docker tag water-potability-app:latest <account_id>.dkr.ecr.us-east-1.amazonaws.com/water-potability-app:latest
    ```

3.  **Push the image**:
    ```powershell
    docker push <account_id>.dkr.ecr.us-east-1.amazonaws.com/water-potability-app:latest
    ```

## Step 3: Deploy with AWS App Runner

1.  Search for **AWS App Runner** in the console.
2.  Click **Create service**.
3.  **Source**: Select **Container registry**.
4.  **Provider**: Select **Amazon ECR**.
5.  **Image URI**: Browse and select the image you just pushed.
6.  **Deployment settings**: Select **Automatic** (deploys every time you push a new image).
7.  **Configuration**:
    *   **Port**: `8000`
    *   **Start command**: Leave blank (it uses the Dockerfile CMD).
8.  Click **Next** -> **Create & deploy**.

## Step 4: Access Your App

Once deployed (takes ~5 mins), App Runner will provide a **Default domain** (e.g., `https://xyz.awsapprunner.com`).

You can now update your `src/streamlit_app.py` to point to this URL instead of `localhost:8000`.

```python
# src/streamlit_app.py
response = requests.post("https://<your-app-runner-url>/predict", json=payload)
```
