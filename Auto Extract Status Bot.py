class extract_bot:
    
    def __init__(self, app_num_input, pp_num_input, dob_num_input, 
                 extract_max_iter, time_delay_sec, music_iter, url):
        
        self.app_num_input = app_num_input
        self.pp_num_input = pp_num_input
        self.dob_num_input = dob_num_input
        self.extract_max_iter = extract_max_iter
        self.extract_time_delay_sec = extract_time_delay_sec
        self.music_iter = music_iter
        self.url = url

    def clear_text(ele):
        """Clear the text in the search bar"""
        ele.send_keys(Keys.CONTROL + "a")
        ele.send_keys(Keys.DELETE)

    def initialize(self):
        """"Initialize the bot"""
        app_num = driver.find_elements_by_xpath("//input[@name='aplno']")[0]
        clear_text(app_num)
        app_num.send_keys(self.app_num_input)

        pp_num = driver.find_elements_by_xpath("//input[@name='ID']")[0]
        clear_text(pp_num)
        pp_num.send_keys(self.pp_num_input)

        dob = driver.find_elements_by_xpath("//input[@name='Birth']")[0]
        clear_text(dob)
        dob.send_keys(self.dob_num_input)

    def clicker():
        """Click on certain elements"""
        time.sleep(2)
        
        #click checkbox
        driver.find_elements_by_xpath("//input[@name='chkpdpa']")[0].click()

        time.sleep(2)

        #click submit
        driver.find_elements_by_xpath("//input[@name='boption']")[0].click()

    def play_music():
        """Play music"""
        display(Audio('Chillstep Ferven - Falling.mp3', autoplay=True))

    def repeat_music(self):
        n = 0
        while n < self.music_iter:

            music()
            time.sleep(218)
            n+=1

    def extract_status(self):
        """Extract application status"""
        app_status = driver.find_elements_by_xpath(".//*[text()='Application under process ']")[0].text
        i = 0

        while i <= self.extract_max_iter:

            if app_status == 'Application under process @':

                try:

                    driver.refresh()
                    app_status = driver.find_elements_by_xpath(".//*[text()='Application under process ']")[0].text
                    print("Iteration (" + str(i) + "): " + app_status)
                    time.sleep(self.extract_time_delay_sec) 
                    i+=1

                    if i == self.extract_max_iter + 1:
                        print("End of the Loop.")
                        #send_simple_message_validate()

                except:
                    wrapper_function(self.extract_max_iter, self.extract_time_delay_sec)
                    time.sleep(self.extract_time_delay_sec)

            else:
                print("Updates Available.")
                repeat_music(self.music_iter);
                #send_simple_message_available()
                break

    def wrapper_function(self):
        """Wrap all functions together"""
        try:
            driver.get(self.url)
            extract_bot.initialize(self)
            clicker()
            extract_bot.extract_status(self)
        except:
            driver.get(self.url)
            extract_bot.initialize(self)
            clicker()
            extract_bot.extract_status(self)
            
            
app_num_input = getpass()
pp_num_input = getpass()
dob_num_input = getpass()
extract_max_iter = 20000
extract_time_delay_sec = 300
music_iter = 20
url = 'url'

extract_bot_obj = extract_bot(app_num_input, pp_num_input, dob_num_input, 
                              extract_max_iter, time_delay_sec, music_iter, url)

extract_bot_obj.wrapper_function()
