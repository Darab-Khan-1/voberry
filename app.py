import streamlit as st
import asyncio
from scrape_docs import main

st.title("📄 Documentation Scraper")

# Input for base URL
base_url = st.text_input("Enter the base URL of the docs site (e.g., https://docs.livekit.io):")

# Input for the company name
company_name = st.text_input("Enter the company name for the assistant (e.g., LiveKit):")

# Store the company name in session state
if company_name:
    st.session_state.company = company_name

# Display current company name
if 'company' in st.session_state:
    st.write(f"Current company name: {st.session_state.company}")

# Define an async function to run the scraping process
async def run_scraping(base_url):
    try:
        # Call the main function asynchronously
        await main(base_url)
        st.success("Scraping completed successfully!")
    except Exception as e:
        st.error(f"Error: {e}")

# Start Scraping button
if st.button("Start Scraping"):
    if base_url.strip():
        st.info("Scraping started... This may take a while.")
        try:
            # Use the current event loop to run the async function
            asyncio.run(run_scraping(base_url))
        except Exception as e:
            st.error(f"Error: {e}")
    else:
        st.warning("Please enter a valid URL.")
