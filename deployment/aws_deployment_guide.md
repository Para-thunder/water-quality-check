# AWS Deployment Guide (EC2 with Docker)

This guide outlines how to deploy your Water Potability App to an **AWS EC2 instance** using Docker.

## Prerequisites

1.  **AWS Account**: You need an active AWS account.
2.  **SSH Client**: Terminal (Mac/Linux) or PuTTY/PowerShell (Windows).

## Step 1: Launch an EC2 Instance

1.  Log in to the AWS Console and search for **EC2**.
2.  Click **Launch Instance**.
3.  **Name**: `WaterPotabilityServer`.
4.  **OS Image**: Select **Ubuntu** (Ubuntu Server 22.04 LTS is recommended).
5.  **Instance Type**: `t3.micro` (Free tier eligible in most regions) or `t2.medium` (Recommended for TensorFlow if you have budget).
    *   *Note: If `t2.micro` is missing, `t3.micro` is the newer, better version and is usually also free-tier eligible.*
6.  **Key Pair**: Create a new key pair (e.g., `my-key.pem`) and download it. **Keep this safe!**
7.  **Network Settings**:
    *   Check **Allow SSH traffic from Anywhere**.
    *   Check **Allow HTTP traffic from the internet**.
    *   Check **Allow HTTPS traffic from the internet**.
8.  Click **Launch Instance**.

## Step 2: Configure Security Group (Open Port 8000)

1.  Go to your EC2 Dashboard -> **Instances**.
2.  Click on your instance ID.
3.  Click the **Security** tab -> Click the **Security Group** link (e.g., `sg-0123...`).
4.  Click **Edit inbound rules**.
5.  Add a new rule:
    *   **Type**: Custom TCP
    *   **Port range**: `8000`
    *   **Source**: `0.0.0.0/0` (Anywhere)
6.  Click **Save rules**.

## Step 3: Connect to Your Instance

1.  Open your terminal (PowerShell).
2.  Navigate to where your key file (`my-key.pem`) is.
3.  Connect using SSH (replace `1.2.3.4` with your EC2 Public IP):
    ```powershell
    ssh -i "my-key.pem" ubuntu@1.2.3.4
    ```

## Step 4: Install Docker on EC2

Run these commands inside your EC2 terminal:

```bash
# Update packages
sudo apt-get update

# Install Docker
sudo apt-get install -y docker.io

# Start Docker
sudo systemctl start docker
sudo systemctl enable docker

# Add user to docker group (avoids using sudo for docker commands)
sudo usermod -aG docker $USER
```
*Log out (`exit`) and log back in for the group change to take effect.*

## Step 5: Deploy the App

You have two options: **Option A (Build on Server)** or **Option B (Pull from Docker Hub/ECR)**. Option A is simpler for a quick start.

### Option A: Clone and Build

1.  **Clone your repository**:
    ```bash
    git clone https://github.com/Para-thunder/water-quality-check.git
    cd water-quality-check
    ```

2.  **Build the Docker image**:
    ```bash
    docker build -t water-app .
    ```
    *(This might take a few minutes)*

3.  **Run the container**:
    ```bash
    docker run -d -p 8000:8000 --name water-app-container water-app
    ```

## Step 6: Access Your App

1.  Find your **Public IPv4 address** in the EC2 console.
2.  Open your browser and go to: `http://<your-public-ip>:8000`
3.  You should see your Water Potability App!

## Maintenance

*   **View Logs**: `docker logs water-app-container`
*   **Stop App**: `docker stop water-app-container`
*   **Update App**:
    ```bash
    git pull
    docker build -t water-app .
    docker stop water-app-container
    docker rm water-app-container
    docker run -d -p 8000:8000 --name water-app-container water-app
    ```

