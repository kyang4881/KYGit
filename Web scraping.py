from requests import get
from requests.exceptions import RequestException  #pip3 install requests
from contextlib import closing
from bs4 import BeautifulSoup

def is_good_response(resp):

    #Returns true if the response seems to be HTML, false otherwise

    content_type = resp.headers['Content-Type'].lower()
    return (resp.status_code == 200 
            and content_type is not None 
            and content_type.find('html') > -1)

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

NPIs = (1316416159, 1669583191, 1841261658)

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


