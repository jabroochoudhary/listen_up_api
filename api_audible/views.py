import numpy as np
from bokeh.plotting import figure, output_file, save
from bokeh.models import FuncTickFormatter, FixedTicker, Range1d, ColumnDataSource, HoverTool
import os
import json
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from django.conf import settings
from random import random as rand
import cvxpy as cp
import pandas as pd
from .models import DFModel

def InterpolationJacobian(tt, tz):
    M = len(tz)
    N = len(tt)
    Ji = np.zeros((M,N))
	#determine a simple relation tt[k] = t0 + k*dt
    t0 = tt[0]
    dt = tt[1]-tt[0]
    for k,t in enumerate(tz):
        nk = (t-t0)/dt
        n0 = int(np.floor(nk))
        if (n0<0) or (n0>=N-1):
            print('out of bounds error')
        w = nk-n0
        Ji[k,n0  ] = 1-w
        Ji[k,n0+1] =   w
    return Ji


def draw_piano(p):
    key_width = 1.0/7.0
    keys_below = 23
    keys_above = 28
    db_offset = 65
    left_edge = -(keys_below+0.5)*key_width
    right_edge = (keys_above+0.5)*key_width
    p.rect(x=(left_edge+right_edge)/2.0, y=db_offset, width=right_edge-left_edge, height=12, fill_color='black')
    for n in range(keys_below+keys_above+1): #white keys
        p.rect(x=(n-keys_below)*key_width, y=db_offset, width=key_width-.01, height=12-0.1, fill_color='white')
    for n in range(keys_below+keys_above): #black keys
        key = (n-keys_below+1) #define as "flats" relative to this key
        k7 = np.mod(key,7)
        if k7!=0 and k7!=3: #exclude C-flat and F-flat
            p.rect(x=(key-0.5)*key_width, y=db_offset-2, width=key_width/2, height=8-0.1, fill_color='black')

def load_regression_sensitivity(user_name):
     loaded_data = np.load(f'reg_files/{user_name}.npy')
     return loaded_data

def ColumnDataSourcep(df):
    pitch = np.array(df['pitch'])
    yy = np.array(df['dB_estimated'])
    frequency = np.array(df['frequency'])
    nx = np.array(df['right_ear'])==1 #logical array, true for right ear
    sourceL = ColumnDataSource(data=dict(x=pitch[~nx], y=yy[~nx], frequency=frequency[~nx], ear=[ "left"]*len(yy[~nx])))
    sourceR = ColumnDataSource(data=dict(x=pitch[ nx], y=yy[ nx], frequency=frequency[ nx], ear=["right"]*len(yy[ nx])))
    return sourceL, sourceR

def audiogram(df,regression_sensitivity, file_name):

    output_file(file_name)
    p = figure(width=1100, height=500, active_scroll="wheel_zoom", x_range=(-4, 6), x_axis_label='pitch', y_axis_label='Audiogram (dB)', y_range=Range1d(start=70, end=-00))
    p.xaxis.ticker = FixedTicker(ticks=[-4,-3,-2,-1,0,1,2,3,4,5,6])
    p.xaxis.formatter = FuncTickFormatter(code="""
		var labels = ['C-4','C-3','C-2','C-1','C','C+1','C+2','C+3','C+4','C+5','C+6'];
	    return labels[tick+4];
	""")
    draw_piano(p)
    pitch_axis = np.arange(-4.00,6.01,1./12.)
    p.line(pitch_axis, regression_sensitivity[:,1], line_width=2, color="red")
    p.line(pitch_axis, regression_sensitivity[:,0], line_width=2, color="navy")
    sourceL, sourceR = ColumnDataSourcep(df)
    p.circle('x', 'y', source=sourceL, size=14, color="navy", alpha=0.4, legend_label="Left")
    p.circle('x', 'y', source=sourceR, size=14, color="red",  alpha=0.4, legend_label="Right")
    nx_current = df['session_start'] == df.iloc[-1]['session_start'] #bold the current session
    sourceL, sourceR = ColumnDataSourcep(df[nx_current])
    p.circle('x', 'y', source=sourceL, size=14, color="navy", alpha=0.7, legend_label="Left")
    p.circle('x', 'y', source=sourceR, size=14, color="red",  alpha=0.7, legend_label="Right")
    tooltips = [
	    ("Pitch", "@x"),
		("dB", "@y"),
		("Ear", "@ear"),
		("Frequency", "@frequency"),
		    # ("Placeholder", "@info2"),
	]
    hover = HoverTool(tooltips=tooltips)
    p.add_tools(hover)
    p.legend.location = "top_left"
    save(p)
    return file_name           


def getSpan(target_db, regression_sensitivity):
		yy = (regression_sensitivity[:,0] + regression_sensitivity[:,1])/2.0
		xx = np.arange(-4.00,6.01,1./12.)
		threshold = target_db
		#note that audiogram is flipped, so rising goes down
		nx_rising = np.where(-1==np.diff((yy > target_db).astype(int)))[0]
		nx_falling = np.where(1==np.diff((yy > target_db).astype(int)))[0]
		if len(nx_rising)==0 or len(nx_falling)==0:
			return np.array([0.0, 0.0])
		nx = [nx_rising[0], nx_falling[-1]]
		span = []
		for idx in nx:
			# Linear interpolation of x values
			x0, x1 = xx[idx], xx[idx + 1]
			y0, y1 = yy[idx], yy[idx + 1]
			x_interp = x0 + (x1 - x0) * (threshold - y0) / (y1 - y0)
			span.append(x_interp)
		return np.array(span)

def get_summary(regression_sensitivity):
    span = getSpan(regression_sensitivity=regression_sensitivity,target_db=40)
    total_span = span[1]-span[0]
    summary = ""
    if total_span>5.0:
        summary = 'Your hearing range is reasonable\n'
        summary += 'Audible span of %.2f octaves (%.2f, %.2f) at 40 dB' % (total_span, span[0], span[1])
    else:
        summary += 'Audible span of %.2f octaves (%.2f, %.2f) at 40 dB' % (total_span, span[0], span[1])
    
    return summary

@csrf_exempt
def get_audiogram(request):
    if request.method == 'POST':
        try:
            data = json.loads(request.body.decode('utf-8'))

            user_name = data.get('user_name')
            df = load_df(user_name=user_name)
            regression_sen = load_regression_sensitivity(user_name=user_name)



            file_name = f"{settings.MEDIA_ROOT}/{user_name}.html"
           
            crated_file_path = audiogram(df=df, file_name=file_name, regression_sensitivity=regression_sen)

            url = crated_file_path
            summary = get_summary(regression_sensitivity=regression_sen)

            return JsonResponse({'url':url,"summary":summary},status=200)

        except json.JSONDecodeError:
            return JsonResponse({'error': 'Invalid JSON data'}, status=400)

    return JsonResponse({'error': 'Invalid request method'}, status=405)

# dddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddd



def InterpolationJacobian(tt, tz):
    M = len(tz)
    N = len(tt)
    Ji = np.zeros((M,N))
    #determine a simple relation tt[k] = t0 + k*dt
    t0 = tt[0]
    dt = tt[1]-tt[0]
    for k,t in enumerate(tz):
        nk = (t-t0)/dt
        n0 = int(np.floor(nk))
        if (n0<0) or (n0>=N-1):
            print('out of bounds error')
        w = nk-n0
        Ji[k,n0  ] = 1-w
        Ji[k,n0+1] =   w
    return Ji

def profile_regression(pitch_axis,df, start_date=None, end_date=None):
    tt = pitch_axis
    nx_valid = np.array(df['error_code']) == 0
    df = df[nx_valid]
    nx_date = [True]*len(df)
    session_time = pd.to_datetime(df['session_start'])
    if (start_date is not None):
        nx_date = nx_date and (session_time >= pd.to_datetime(start_date))
    if end_date is not None:
        nx_date = nx_date and (session_time <= pd.to_datetime(end_date))
    if len(df)>10: #enough data to exclude the 1900 starters
        nx_date = nx_date and (session_time >= pd.to_datetime('2000'))
    df = df[nx_date]
    tz = np.array(df['pitch'])
    z_meas = np.array(df['dB_estimated'])
    n = len(tt)
    left = cp.Variable(n)
    right = cp.Variable(n)
    reaction_time = cp.Variable(1)
    nx_right = np.array(df['right_ear'])==1
    D = np.diag([-1]*(n-1) + [0], k=0) + np.diag([1]*(n-1), k=1)
    D2 = D[:-1, :] @ D
    l1_reg = cp.norm(D2 @ left, 1)    + cp.norm(D2 @ right, 1)
    l2_reg = cp.norm(D2 @ left, 2)**2 + cp.norm(D2 @ right, 2)**2
    Ji = InterpolationJacobian(tt, tz[~nx_right])
    l2_data_L = cp.norm(Ji @ left  - z_meas[~nx_right], 2)**2
    l1_data_L = cp.norm(Ji @ left  - z_meas[~nx_right], 1)
    # print(tt)
    Ji = InterpolationJacobian(tt, tz[nx_right])
    l2_data_R = cp.norm(Ji @ right - z_meas[nx_right], 2)**2
    l1_data_R = cp.norm(Ji @ right - z_meas[nx_right], 1)
    rt_term = cp.norm(reaction_time - reaction_time, 2)**2
    bind_term = cp.norm(left - right, 1)
    data_term = l2_data_L + l1_data_L + l2_data_R + l1_data_R + rt_term
    objective = cp.Minimize(data_term + 5*l1_reg + 3*l2_reg + 3*bind_term)
    prob = cp.Problem(objective)
    prob.solve(solver=cp.SCS)
    regression_sensitivity = np.column_stack( (left.value, right.value) )
    return regression_sensitivity # regression_reaction_time = reaction_time.value[0]


def query_sensitivity(pitch_axis,pitch,regression_sensitivity):
	if not ((pitch_axis[0] <= pitch) and (pitch <= pitch_axis[-1]) ):
		print('query_sensitivity out-of-bounds error')
		print('%.2f %.2f %.2f' % (pitch_axis[0], pitch_axis[1], pitch) )
	Ji = InterpolationJacobian(pitch_axis, [pitch])
	sensitivity_query = Ji @ regression_sensitivity
	return sensitivity_query[0]



def freq_gen():
    pitch_range = [-3.5, 5.5]
    middle_C = 261.63
    pr = pitch_range
    pitch = pr[0] + (pr[1]-pr[0])*rand()
    frequency = middle_C * 2**pitch
    return pitch, frequency

def play_crescendo(df):
    db_per_sec = 20.0/np.sqrt(len(df)) # rate at which the volume increases
    db_per_sec = np.clip(db_per_sec,2.0,5.0) # returnable
    wait_time = 4.0
    reaction_time = 0.8
    pitch_axis = np.arange(-4.00,6.01,1./12.)
    # period = period
    pitch, frequency = freq_gen()  #returnable
    regression_sensitivity = profile_regression(df=df,pitch_axis=pitch_axis) 
    sensitivity_query = query_sensitivity(pitch_axis ,pitch, regression_sensitivity ) #returnable
    T_wait = wait_time + reaction_time
    db_start = sensitivity_query - T_wait*db_per_sec #returnable
    return pitch, frequency, db_start, regression_sensitivity

def df_log(db_start,db_per_sec,reaction_time,sensitivity,session_str,idx,ear_idx,response_time,frequency,pitch,error_code,time_seconds,df):
	# response_time = time.time()-self.tic
	# sensitivity = db_start +  db_per_sec * (response_time- self.reaction_time)
	# time_seconds = (datetime.now() - session_time).total_seconds()
	new_data = {'session_start': [session_str], 
	            'time_seconds': [time_seconds], 
	            'idx': [idx], 
	            'right_ear': [ear_idx], 
	            'frequency': [frequency], 
	            'pitch': [pitch], 
	            'dB_estimated': [sensitivity], 
	            'dB_start': [db_start], 
	            'dB_per_sec': [db_per_sec], 
	            'time_detected': [response_time], 
	            'error_code': [error_code]}
	if idx > 0: #ignore the first sample as a warm-up
		df = pd.concat([df, pd.DataFrame(new_data)], ignore_index=True)
	return df
    

def load_df(user_name):
    if os.path.exists(f"csv_files/{user_name}.csv"):
        df = pd.read_csv(f"csv_files/{user_name}.csv")
        return df
    else:
        df = pd.read_csv("csv_files/new_user.csv")
        df.to_csv(f"csv_files/{user_name}.csv", index=False)
        df = pd.read_csv(f"csv_files/{user_name}.csv")
        return df


def write_df(df,user_name):
     df.to_csv(f"csv_files/{user_name}.csv", index=False)

def save_regression_sensitivity(regression_sensitivity,user_name):
     np.save(f'reg_files/{user_name}.npy', regression_sensitivity)

def load_regression_sensitivity(user_name):
     loaded_data = np.load(f'reg_files/{user_name}.npy')
     return loaded_data

# Example usage:
# user_name = "example_user"
# df = load_df(user_name)
# print(df)

@csrf_exempt
def regression_profile(request):
    if request.method == 'POST':
        try:
            data = json.loads(request.body.decode('utf-8'))

            user_name = data.get('user_name')
            name = data.get('name')
            dob = data.get('dob')
            device = data.get('device')

            pitch = data.get('pitch')
            frequency = data.get('frequency')
            right_ear = data.get('right_ear')
            reaction_time = data.get('reaction_time')
            session_time = data.get('session_time')
            db_start = data.get('db_start')
            db_per_sec = data.get('db_per_sec')
            sensitivity = data.get('sensitivity')
            time_seconds = data.get('time_seconds')
            error_code = data.get('error_code')
            response_time = data.get('response_time')
            idx = data.get('idx')
  
            df = load_df(user_name)
            if pitch is None:
                pitch, frequency, db_start, regression_sensitivity = play_crescendo(df=df)
                save_regression_sensitivity(regression_sensitivity=regression_sensitivity,user_name=user_name)
                # print(db_start)
                dt = {"pitch":pitch,"frequency":frequency,"db_start":db_start[0],"msg":"new data generated","status":True}
                return JsonResponse(data=dt, status=200)
            else:
                DFModel.objects.create(
                session_start=session_time,
                time_seconds=time_seconds,
                idx=idx,
                right_ear=right_ear,
                frequency=frequency,
                pitch=pitch,
                dB_estimated=sensitivity,
                dB_start=db_start,
                dB_per_sec=db_per_sec,
                time_detected=response_time,
                error_code=error_code,
                name=name,
                user_id=user_name,
                dob=dob,
                device = device)
            
                # print(df)
                # print("###################################################")
                df = df_log(df=df,db_per_sec=db_per_sec,db_start=db_start,
                            ear_idx=right_ear,error_code=error_code,frequency=frequency,
                            idx=idx, pitch=pitch,reaction_time=reaction_time,response_time=response_time,
                            sensitivity=sensitivity,session_str=session_time,time_seconds=time_seconds)
                write_df(df=df, user_name=user_name)
                df = load_df(user_name)
                # print(df)
                # print("###################################################")
                pitch, frequency, db_start, regression_sensitivity = play_crescendo(df=df)
                
                save_regression_sensitivity(regression_sensitivity=regression_sensitivity,user_name=user_name)
               
                dt = {"pitch":pitch,"frequency":frequency,"db_start":db_start[0],"msg":"Log added and new data generated","status":True}
                return JsonResponse(data=dt, status=200)
                

        except json.JSONDecodeError:
            return JsonResponse({'error': 'Invalid JSON data types'}, status=400)

    return JsonResponse({'error': 'Invalid request method'}, status=405)


def delete_static_files(user_name):
    file_path = f'reg_files/{user_name}.npy'
    if os.path.exists(file_path):
        os.remove(file_path)

    file_path = f"csv_files/{user_name}.csv"
    if os.path.exists(file_path):
        os.remove(file_path)

    file_path = f"media/{user_name}.html"
    if os.path.exists(file_path):
        os.remove(file_path)

    return True


@csrf_exempt
def reset_user(request):
    if request.method == 'POST':
        try:
            data = json.loads(request.body.decode('utf-8'))

            user_name = data.get('user_name')
            
            if user_name:
                # Delete rows where user_name matches
                DFModel.objects.filter(user_id=user_name).delete()
                delete_static_files(user_name=user_name)


                return JsonResponse({'message': f'Rows with user_name {user_name} deleted successfully!'}, status=200)
            else:
                return JsonResponse({'error': 'user_name not provided in the request.'}, status=400)

        except json.JSONDecodeError:
            return JsonResponse({'error': 'Invalid JSON data types'}, status=400)

    return JsonResponse({'error': 'Invalid request method'}, status=405)