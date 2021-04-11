"""Main module for the streamlit app"""
import streamlit as st
import pages.upload
import pages.search

PAGES = {
    "Search": pages.search,
    "Upload": pages.upload,
}


def main():
    """Main function of the App"""
    st.sidebar.title("Navigation")
    selection = st.sidebar.radio("Go to", list(PAGES.keys()))

    page = PAGES[selection]

    page.write()

if __name__ == "__main__":
    main()