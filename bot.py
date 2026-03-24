import requests

url = "https://sims.efrei.educentre.fr/api/v1/startups/a4f3bc06-9fbe-4599-80d3-9bbfa6bf7b2e/status"

headers = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/122.0.0.0 Safari/537.36"
    )
}

response = requests.get(url, headers=headers, timeout=10)

print(response.status_code)
print(response.json())