mkdir -p ~/.streamlit

cat << EOF > ~/.streamlit/config.toml
[server]
headless = true
port = 8501
enableCORS = false
enableXsrfProtection = false

[browser]
gatherUsageStats = false
serverAddress = "0.0.0.0"
serverEnableCORS = false
serverEnableXsrfProtection = false

[theme]
primaryColor = "#1E3A8A"
backgroundColor = "#FFFFFF"
secondaryBackgroundColor = "#F0F2F6"
textColor = "#1A1A1A"
font = "sans serif"
EOF
