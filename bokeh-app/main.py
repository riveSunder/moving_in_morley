import os


import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from carle.env import CARLE
from carle.mcl import SpeedDetector
from game_of_carle.agents.toggle import Toggle

import bokeh
from bokeh.io import curdoc
import bokeh.io as bio
from bokeh.io import output_notebook, show
from bokeh.plotting import figure

from bokeh.layouts import column, row
from bokeh.models import TextInput, Button, Paragraph
from bokeh.models import ColumnDataSource
from bokeh.events import DoubleTap, Tap

agent =  Toggle()
policy_list = []
directory_list = os.listdir("policies/")

for filename in directory_list:
    
    if "Toggle" in filename and "glider" in filename:
        
        policy_list.append(os.path.join("policies", filename))
        
# instantiate CARLE with a speed detection wrapper
env = CARLE(height=128, width=128)
env = SpeedDetector(env)

# set Move/Morley rules
my_rules = "B368/S245"

agent.set_params(np.load(policy_list[0]))

print(f"{len(policy_list)} Toggle glider policies found.")

env.rules_from_string(my_rules)

global obs
obs = env.reset()

p = figure(plot_width=3*256, plot_height=3*256, title="CA Universe")
p_plot = figure(plot_width=int(2.5*256), plot_height=int(2.5*256), title="'Reward'")

global my_period
global number_agents
global agent_number

agent_number = 0
number_agents = len(policy_list)
my_period = 512

source = ColumnDataSource(data=dict(my_image=[obs.squeeze().cpu().numpy()]))
source_plot = ColumnDataSource(data=dict(x=np.arange(1), y=np.arange(1)*0))

img = p.image(image='my_image',x=0, y=0, dw=256, dh=256, palette="Magma256", source=source)
line_plot = p_plot.line(line_width=3, color="firebrick", source=source_plot)

button_go = Button(sizing_mode="stretch_width", label="Run >")     
button_slower = Button(sizing_mode="stretch_width",label="<< Slower")
button_faster = Button(sizing_mode="stretch_width",label="Faster >>")
button_reset_prev_agent = Button(sizing_mode="stretch_width",label="Reset w/ Prev. Agent")
button_reset_this_agent = Button(sizing_mode="stretch_width",label="Reset w/ This Agent")
button_reset_next_agent = Button(sizing_mode="stretch_width",label="Reset w/ Next Agent")

button_agent_switch = Button(sizing_mode="stretch_width", label="Turn Agent Off")

message = Paragraph()

def update():
        global obs
        global stretch_pixel
        global action
        global agent_on
        global my_step
        global rewards
        global agent_number
        
        obs, r, d, i = env.step(action)
        rewards = np.append(rewards, r.cpu().numpy().item())
        if agent_on:
            action = agent(obs) 
        else:
            action = torch.zeros_like(action)
            
        #padded_action = stretch_pixel/2 + env.action_padding(action).squeeze()
        padded_action = stretch_pixel/2 + env.inner_env.action_padding(action).squeeze()
        
        my_img = (padded_action*2 + obs.squeeze()).cpu().numpy()
        my_img[my_img > 3.0] = 3.0
        (padded_action*2 + obs.squeeze()).cpu().numpy()
        new_data = dict(my_image=[my_img])
        
        #new_line = dict(x=np.arange(my_step+2), y=rewards)
        new_line = dict(x=[my_step], y=[r.cpu().numpy().item()])
        
        source.stream(new_data, rollover=1)
        source_plot.stream(new_line, rollover=2000)
        
        my_step += 1
        message.text = f"agent {agent_number}, step {my_step}, reward: {r.item()} \n"\
                f"{policy_list[agent_number]}"
    
def go():
   
    if button_go.label == "Run >":
        my_callback = curdoc().add_periodic_callback(update, my_period)
        button_go.label = "Pause"
        
    else:
        curdoc().remove_periodic_callback(curdoc().session_callbacks[0])
        button_go.label = "Run >"

def faster():
    global my_period
    my_period = max([my_period / 2, 1])
    go()
    go()
    
def slower():
    global my_period
    my_period = min([my_period * 2, 8192])
    go()
    go()

def reset_this_agent():
    global obs
    global stretch_pixel
    global my_step
    global rewards
    global agent_number
    global number_agents
    
    my_step = 0
    
    obs = env.reset()        
    agent.reset()
            
    stretch_pixel = torch.zeros_like(obs).squeeze()
    stretch_pixel[0,0] = 3
    new_data = dict(my_image=[(stretch_pixel + obs.squeeze()).cpu().numpy()])
    rewards = np.array([0])
    
    new_line = dict(x=[my_step], y=[0])
    
    source_plot.stream(new_line, rollover=1)
    source.stream(new_data, rollover=8)
 
def reset_next_agent():
   
    global obs
    global stretch_pixel
    global my_step
    global rewards
    global agent_number
    global number_agents
    
    my_step = 0
    
    obs = env.reset()        
            
    stretch_pixel = torch.zeros_like(obs).squeeze()
    stretch_pixel[0,0] = 3
    new_data = dict(my_image=[(stretch_pixel + obs.squeeze()).cpu().numpy()])
    rewards = np.array([0])
    
    new_line = dict(x=[my_step], y=[0])
    
    source_plot.stream(new_line, rollover=1)
    source.stream(new_data, rollover=8)
    
    agent_number = (agent_number + 1) % number_agents
    
    agent.set_params(np.load(policy_list[agent_number]))
    agent.reset()
    
    message.text = f"reset with agent {agent_number}"
        
def reset_prev_agent():
   
    global obs
    global stretch_pixel
    global my_step
    global rewards
    global agent_number
    global number_agents
    
    my_step = 0
    
    obs = env.reset()        
            
    stretch_pixel = torch.zeros_like(obs).squeeze()
    stretch_pixel[0,0] = 3
    new_data = dict(my_image=[(stretch_pixel + obs.squeeze()).cpu().numpy()])
    rewards = np.array([0])
    
    new_line = dict(x=[my_step], y=[0])
    
    source_plot.stream(new_line, rollover=1)
    source.stream(new_data, rollover=8)
    
    agent_number = (agent_number - 1) % number_agents
    
    agent.set_params(np.load(policy_list[agent_number]))
    agent.reset()
    
    message.text = f"reset with agent {agent_number}"

def human_toggle(event):
    global action

    coords = [np.round(env.height*event.y/256-0.5), np.round(env.width*event.x/256-0.5)]
    offset_x = (env.height - env.action_height) / 2
    offset_y = (env.width - env.action_width) / 2

    print(offset_x, coords[0])
    coords[0] = coords[0] - offset_x
    coords[1] = coords[1] - offset_y

    print(offset_x, coords)
    coords[0] = np.uint8(np.clip(coords[0], 0, env.action_height-1))
    coords[1] = np.uint8(np.clip(coords[1], 0, env.action_height-1))

    action[:, :, coords[0], coords[1]] = 1.0 * (not(action[:, :, coords[0], coords[1]]))

    print(offset_x, coords[0])
    #padded_action = stretch_pixel/2 + env.action_padding(action).squeeze()
    padded_action = stretch_pixel/2 + env.inner_env.action_padding(action).squeeze()

    my_img = (padded_action*2 + obs.squeeze()).cpu().numpy()
    my_img[my_img > 3.0] = 3.0
    (padded_action*2 + obs.squeeze()).cpu().numpy()
    new_data = dict(my_image=[my_img])

    source.stream(new_data, rollover=8)

def clear_toggles():
    global action

    if button_go.label == "Pause":

        action *= 0
        doc.remove_periodic_callback(doc.session_callbacks[0])
        button_go.label = "Run >"

        #padded_action = stretch_pixel/2 + env.action_padding(action * 0).squeeze()
        padded_action = stretch_pixel/2 + env.inner_env.action_padding(action).squeeze()

        my_img = (padded_action*2 + obs.squeeze()).cpu().numpy()
        my_img[my_img > 3.0] = 3.0
        (padded_action*2 + obs.squeeze()).cpu().numpy()
        new_data = dict(my_image=[my_img])

        source.stream(new_data, rollover=8)
    else:
        doc.add_periodic_callback(update, my_period)
        button_go.label = "Pause"

def agent_on_off():
    global agent_on
    
    if button_agent_switch.label == "Turn Agent Off":
        agent_on = False
        button_agent_switch.label = "Turn Agent On"
            
    else:
        agent_on = True
        button_agent_switch.label = "Turn Agent Off"

global agent_on
agent_on = True
global action
action = torch.zeros(1, 1, env.action_height, env.action_width)

reset_this_agent()

p.on_event(Tap, human_toggle)
p.on_event(DoubleTap, clear_toggles)

button_reset_prev_agent.on_click(reset_prev_agent)
button_reset_this_agent.on_click(reset_this_agent)
button_reset_next_agent.on_click(reset_next_agent)

button_go.on_click(go)
button_faster.on_click(faster)
button_slower.on_click(slower)
button_agent_switch.on_click(agent_on_off)

display_layout = row(p, p_plot)
control_layout = row(button_slower, button_go, button_faster)
toggle_toggle_agent_layout = row(button_reset_prev_agent, \
        button_reset_this_agent, \
        button_reset_next_agent)

message_layout = row(message)
agent_toggle_layout = row(button_agent_switch)

curdoc().add_root(display_layout)
curdoc().add_root(control_layout)
curdoc().add_root(toggle_toggle_agent_layout)
curdoc().add_root(message_layout)
curdoc().add_root(agent_toggle_layout)
