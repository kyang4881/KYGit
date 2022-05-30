def clear_text(ele):
    """Clear the text in the search bar"""
    
    ele.send_keys(Keys.CONTROL + "a")
    ele.send_keys(Keys.DELETE)
    
def initialize(app_num_input, pp_num_input, dob_num_input):
    """Initialize the bot"""
    
    app_num = driver.find_elements_by_xpath("//input[@name='aplno']")[0]
    clear_text(app_num)
    app_num.send_keys(app_num_input)

    pp_num = driver.find_elements_by_xpath("//input[@name='ID']")[0]
    clear_text(pp_num)
    pp_num.send_keys(pp_num_input)

    dob = driver.find_elements_by_xpath("//input[@name='Birth']")[0]
    clear_text(dob)
    dob.send_keys(dob_num_input)
    
def clicker():
    """Click on certain elements"""
    
    #click checkbox
    driver.find_elements_by_xpath("//input[@name='chkpdpa']")[0].click()
    
    #click submit
    driver.find_elements_by_xpath("//input[@name='boption']")[0].click()
    
def beep():
    """Play music"""
    
    display(Audio('Chillstep Ferven - Falling.mp3', autoplay=True))

def extract_status(max_iter=10, time_delay_sec=2):
    """Extract application status"""
    
    app_status = driver.find_elements_by_xpath(".//*[text()='Application under process ']")[0].text
    i = 0
    
    while i <= max_iter:
        
        if app_status == 'Application under process @':
            
            try:
                
                driver.refresh()
                app_status = driver.find_elements_by_xpath(".//*[text()='Application under process ']")[0].text
                print("Iteration (" + str(i) + "): " + app_status)
                time.sleep(time_delay_sec) 
                i+=1
                
                if i == max_iter + 1:
                    print("Reached the End of the Loop!")
                    #send_simple_message_validate()
            
            except:
                wrapper_function()
                time.sleep(time_delay_sec)
            
        else:
            print("Updates Available!")
            beep();
            #send_simple_message_available()
            break
        
def wrapper_function(max_iter_given, time_delay_sec_given):
    """Wrap all functions together"""
    
    driver.get('Input URL')
    app_num_input = ''
    pp_num_input = ''
    dob_num_input = ''
    
    initialize(app_num_input, pp_num_input, dob_num_input)
    clicker()
    extract_status(max_iter = max_iter_given, time_delay_sec=time_delay_sec_given)

    
    
#driver = webdriver.Chrome(ChromeDriverManager().install())
#wrapper_function(max_iter_given=20000, time_delay_sec_given=300)
