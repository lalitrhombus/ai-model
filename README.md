commands to depoloy:

gcloud builds submit --tag gcr.io/main-web-app-289309/api-anurag

gcloud run deploy --image gcr.io/main-web-app-289309/api-anurag --platform managed
