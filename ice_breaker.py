from dotenv import load_dotenv
from langchain.chains import LLMChain
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from agents.linkedin_lookup_agent import lookup as linkedin_lookup_agent
from agents.twitter_lookup_agent import lookup as twitter_lookup_agent
from third_parties.linkedin import scrape_linkedin_profile
from third_parties.twitter import scrape_user_tweets
from output_parsers import summary_parser


def ice_break_with(name: str) -> str:
    linkedin_profile_url = linkedin_lookup_agent(name=name)
    linkedin_data = scrape_linkedin_profile(linkedin_profile_url=linkedin_profile_url, mock=True)
    
    twitter_username = twitter_lookup_agent(name=name)
    tweets = scrape_user_tweets(username=twitter_username, mock=True)
    
    summary_template = """
        given the information about a person from linkedin {information},
        and their latest twitter posts {twitter_posts} I want you to create:
        1. A short summary
        2. two interesting facts about them 

        Use both information from twitter and Linkedin
        \n{format_instructions}
    """
    
    summary_prompt_template = PromptTemplate(
        input_valriables=['information','twitter_posts'],
        template=summary_template,
        partial_variables={"format_instructions": summary_parser.get_format_instructions()},
    )
    
    llm = ChatOpenAI(temperature=0, model_name='gpt-3.5-turbo')

    # chain = LLMChain(llm=llm, prompt=summary_prompt_template)
    chain = summary_prompt_template | llm | summary_parser
    
    response = chain.invoke(input={"information": linkedin_data, "twitter_posts": tweets})

    return response


if __name__ == '__main__':
    load_dotenv() # load environment variables dynamically
    response = ice_break_with(name="Eden Marco")
    print(response)