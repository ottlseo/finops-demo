#!/bin/bash

# Update packages
sudo apt-get update -y

# Install required packages
sudo apt-get install -y ec2-instance-connect
sudo apt-get install -y git
sudo apt-get install -y python3-pip
sudo apt-get install -y python3-venv
sudo apt-get install -y jq
sudo apt-get install -y unzip

# Install AWS CLI v2
curl "https://awscli.amazonaws.com/awscli-exe-linux-x86_64.zip" -o "awscliv2.zip"
unzip awscliv2.zip
sudo ./aws/install
rm -rf aws awscliv2.zip

# Clone repository
cd /home/ubuntu
sudo git clone https://github.com/ottlseo/finops-demo.git

# Create virtual environment
sudo python3 -m venv --copies /home/ubuntu/my_env
sudo chown -R ubuntu:ubuntu /home/ubuntu/my_env
source /home/ubuntu/my_env/bin/activate

# Install dependencies
cd finops-demo
sudo git checkout demo/samsung

cd application
sudo apt install -y cargo
pip3 install -r requirements.txt

# Create systemd service
sudo sh -c "cat <<EOF > /etc/systemd/system/streamlit.service
[Unit]
Description=Streamlit App
After=network.target

[Service]
User=ubuntu
Environment='AWS_DEFAULT_REGION=us-west-2'
WorkingDirectory=/home/ubuntu/finops-demo/application
ExecStartPre=/bin/bash -c 'sudo iptables -t nat -A PREROUTING -p tcp --dport 80 -j REDIRECT --to-port 8501'
ExecStart=/bin/bash -c 'source /home/ubuntu/my_env/bin/activate && streamlit run app.py --server.port 8501'
Restart=always

[Install]
WantedBy=multi-user.target
EOF"

# Reload systemd daemon and start the service
sudo systemctl daemon-reload
sudo systemctl enable streamlit
sudo systemctl start streamlit
