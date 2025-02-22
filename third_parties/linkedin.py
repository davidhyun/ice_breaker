import os
import requests
import json
from dotenv import load_dotenv

load_dotenv()


def scrape_linkedin_profile(linkedin_profile_url: str, mock: bool=False):
    """Scrape information from LinkedIn profiles
    Manually scrape the information from the LinkedIn profile
    """

    if mock:
        # Use request with Eden Marco's pre-saved gist(json)
        linkedin_profile_url = "https://api.github.com/gists/1c8fd9626d35844959f53425b8e18f23"
        response = requests.get(
            linkedin_profile_url,
            timeout=10
        )
    else:
        # Use proxycurl (cost generated)
        api_endpoint = "https://nubela.co/proxycurl/api/v2/linkedin"
        headers = {"Authorization": f'Bearer {os.environ.get("PROXYCURL_API_KEY")}'}
        response = requests.get(
            api_endpoint,
            params={"url": linkedin_profile_url},
            headers=headers,
            timeout=10
        )

    data = response.json()

    # minimize feed tokens
    data = {
        k: v
        for k, v in data.items()
        if v not in ([], "", '', None)
        and k not in ["people_also_viewed","certifications"]
    }
    if data.get("groups"):
        for group_dict in data.get("groups"):
            group_dict.pop("profile_pic_url")

    return data    


if __name__ == "__main__":
    linkedin_profile = scrape_linkedin_profile(linkedin_profile_url="https://www.linkedin.com/in/eden-marco/", mock=True)
    print(linkedin_profile)