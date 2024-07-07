from typing import Tuple
from dotenv import load_dotenv
from langchain.chains import LLMChain
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from agents.linkedin_lookup_agent import lookup as linkedin_lookup_agent
from agents.twitter_lookup_agent import lookup as twitter_lookup_agent
from third_parties.linkedin import scrape_linkedin_profile
from third_parties.twitter import scrape_user_tweets
from output_parsers import summary_parser, Summary


def ice_break_with(name: str) -> Tuple[Summary, str]:
    linkedin_profile_url = linkedin_lookup_agent(name=name)
    linkedin_data = scrape_linkedin_profile(linkedin_profile_url=linkedin_profile_url, mock=True)
    
    twitter_username = twitter_lookup_agent(name=name)
    tweets = scrape_user_tweets(username=twitter_username, mock=True)
    
    summary_template = """
        given the information about a person from linkedin {profile},
        and their latest twitter posts {twitter_posts} I want you to create:
        1. A short summary
        2. two interesting facts about them 

        Use both information from twitter and Linkedin
        \n{format_instructions}
    """
    
    summary_prompt_template = PromptTemplate(
        input_valriables=['profile','twitter_posts'],
        template=summary_template,
        partial_variables={"format_instructions": summary_parser.get_format_instructions()},
    )
    
    llm = ChatOpenAI(temperature=0, model_name='gpt-3.5-turbo')

    # chain = LLMChain(llm=llm, prompt=summary_prompt_template)
    chain = summary_prompt_template | llm | summary_parser
    
    information = chain.invoke(input={"profile": linkedin_data, "twitter_posts": tweets})
    # profile_pic_url = linkedin_data.get("profile_pic_url")
    profile_pic_url = "https://media.licdn.com/dms/image/C4D03AQGlv35ItbkHBw/profile-displayphoto-shrink_800_800/0/1610187870291?e=1726099200&v=beta&t=a4O5FVfWWKavlCgkJCcJdzNaIhDs_Cs_M4-TA0xeONI"
    
    return information, profile_pic_url


if __name__ == '__main__':
    load_dotenv() # load environment variables dynamically
    result = ice_break_with(name="Eden Marco Udemy")
    print(result)