#!/bin/bash

# Simple script to copy environment template to .env file

if [ -f ".env" ]; then
    echo "⚠️  .env file already exists!"
    echo "Do you want to overwrite it? (y/N)"
    read -r response
    if [[ ! "$response" =~ ^[Yy]$ ]]; then
        echo "Cancelled."
        exit 0
    fi
fi

if [ -f "env.template" ]; then
    cp env.template .env
    echo "✅ Created .env file from env.template"
    echo ""
    echo "📝 Please edit .env file and update these values:"
    echo "   • MISTRAL_API_KEY: Get from https://mistral.ai/"
    echo "   • QDRANT_URL: Your Qdrant Cloud cluster URL"
    echo "   • QDRANT_API_KEY: Your Qdrant Cloud API key"
    echo ""
    echo "💡 To set up Qdrant Cloud:"
    echo "   1. Sign up at https://cloud.qdrant.io (free tier)"
    echo "   2. Create a cluster"
    echo "   3. Copy the cluster URL and API key"
else
    echo "❌ env.template file not found!"
    echo "Make sure you're in the learnd project directory."
    exit 1
fi
