import PySimpleGUI as sg

def block_focus(window):
    for key in window.key_dict:    # Remove dash box of all Buttons
        element = window[key]
        if isinstance(element, sg.Button):
            element.block_focus()

def popup_drone_ip(drone_ip):
    l = []
    l.append('Label \t  |  IP  ')
    l.append('LM-01-T \t|  192.168.41.246')
    l.append('LM-01   \t\t|  192.168.41.17')
    l.append('LM-01b  \t|  192.168.41.15')
    l.append('LM-02-T \t|  192.168.41.247')
    l.append('LM-02   \t\t|  192.168.41.20')
    l.append('LM-02b  \t|  192.168.41.22')
    l.append('LM-03-T \t|  192.168.41.248')
    l.append('LM-03   \t\t|  192.168.41.24')
    l.append('LM-03b  \t|  192.168.41.19')
    l.append('LM-04-T \t|  192.168.41.249')
    l.append('LM-04   \t\t|  192.168.41.251')
    l.append('LM-04b  \t|  192.168.41.16')
    l.append('LM-05-T \t|  192.168.41.250')
    l.append('LM-05   \t\t|  192.168.41.23')
    l.append('LM-05b  \t|  192.168.41.21')
    l.append('None    \t\t|  192.168.41.18')
    l.append('None    \t\t|  192.168.41.252')
    l.append('Tello Ad-Hoc \t\t|  192.168.10.1')


    col_layout = [[sg.Button('Connect', key='OK', font=("Helvetica", 14))]]
    layout = [
        [sg.Text("What is your team drone IP:\n", font=("Helvetica 14 bold"))],
        [sg.Text("The previous IP was: {}\n".format(drone_ip), font=("Helvetica 14"))],
        [sg.Listbox(l, size=(30, 10), key='-IP-', font=("Helvetica", 14), select_mode='LISTBOX_SELECT_MODE_SINGLE')],
        [sg.Column(col_layout, expand_x=True, element_justification='center')],
    ]
    window = sg.Window("Drone IP", layout, use_default_focus=False, finalize=True, modal=True, 
                    #    icon = cv2.imencode('.ppm', cv2.imread('../images/icons/HumanLoop.png'))[1].tobytes(),
                       )
    block_focus(window)
    event, values = window.read()
    window.close()
    new_ip = values['-IP-'][0].split('|  ')[-1]
    return new_ip if event == 'OK' else None

