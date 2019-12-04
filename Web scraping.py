from requests import get
from requests.exceptions import RequestException  #pip3 install requests
from contextlib import closing
from bs4 import BeautifulSoup

def simple_get(url):
    """
    Attempts to get the content at `url` by making an HTTP GET request.
    If the content-type of response is some kind of HTML/XML, return the
    text content, otherwise return None.
    """
    try:
        with closing(get(url, stream=True)) as resp:
            if is_good_response(resp):
                return resp.content
            else:
                return None

    except RequestException as e:
        log_error('Error during requests to {0} : {1}'.format(url, str(e)))
        return None


def is_good_response(resp):
    """
    Returns True if the response seems to be HTML, False otherwise.
    """
    content_type = resp.headers['Content-Type'].lower()
    return (resp.status_code == 200
            and content_type is not None
            and content_type.find('html') > -1)


url = "https://www.hipaaspace.com/medical_billing/coding/national_provider_identifier/codes/npi_1841261658.aspx"

def get_names(url):
    """
    Downloads the page where the list of mathematicians is found
    and returns a list of strings, one per mathematician
    """
    response = simple_get(url)

    if response is not None:
        html = BeautifulSoup(response, 'html.parser')
        names = set()
        for dd in html.select('dd'):
            for name in dd.text.split('\n'):
                if len(name) > 0:
                    names.add(name.strip())
        return list(names)

    # Raise an exception if we failed to get any data from the url
    raise Exception('Error retrieving contents at {}'.format(url))


def get_names(url):
    """
    Downloads the page where the list of mathematicians is found
    and returns a list of strings, one per mathematician
    """
    #url = 'http://www.fabpedigree.com/james/mathmen.htm'
    response = simple_get(url)

    if response is not None:
        html = BeautifulSoup(response, 'html.parser')
        names = set()
        for dd in html.select('dd'):
            for name in dd.text.split('\n'):
                if len(name) > 0:
                    names.add(name.strip())
        return list(names)

    # Raise an exception if we failed to get any data from the url
    raise Exception('Error retrieving contents at {}'.format(url))


NPIs = (1316416159, 1669583191, 1841261658)

import timeit

def get_pname(NPIs):
    """
    Return the provider names of the given NPI numbers
    """
    
    names = []

    for npi in range(len(NPIs)):
        url = "https://www.hipaaspace.com/medical_billing/coding/national_provider_identifier/codes/npi_" + str(NPIs[npi]) + ".aspx"
        response = simple_get(url)

        if response is not None:
            html = BeautifulSoup(response, 'html.parser')
            first_bar_index = html.title.string.index('|')
            last_bar_index = html.title.string.index('|') + html.title.string[html.title.string.index('|') + 1: len(html.title.string)].index('|') + 1
            names.append(html.title.string[(first_bar_index + 2) : (last_bar_index - 1)])
    return names

get_pname(NPIs)


