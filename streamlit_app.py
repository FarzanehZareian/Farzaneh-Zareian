import streamlit as st
from pycaret.regression import load_model, predict_model
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from annotated_text import annotated_text
import base64
from PIL import Image
import os
import gmspy as gm
from io import StringIO

img = Image.open('steel_structure.jpg')
st.set_page_config(page_title="Peak Dynamic Responses of Steel Moment Frames", page_icon = img)

hide_streamlit_style = """
            <style>
            #MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            </style>
            """
st.markdown(hide_streamlit_style, unsafe_allow_html=True) 

def set_bg_hack(main_bg):
    '''
    A function to unpack an image from root folder and set as bg.
    
    Returns
    -------
    The background.
    '''
    main_bg_ext = "jpg"
        
    st.markdown(
         f"""
         <style>
         .stApp {{
             background: url(data:image/{main_bg_ext};base64,{base64.b64encode(open(main_bg, "rb").read()).decode()});
             background-size: cover}}
         </style>
         """,
         unsafe_allow_html=True)
    
set_bg_hack('theme.jpg')

st.markdown("""
<style>
    [data-testid=stSidebar] {
        background-color: white;
    }
</style>
""", unsafe_allow_html=True)


def processNGAfile(uploaded_file, scalefactor=None):
    '''
    This function process acceleration history for NGA data file (.AT2 format)
    to a single column value and return the total number of data points and 
    time iterval of the recording.
    Parameters:
    ------------
    filepath : string (location and name of the file)
    scalefactor : float (Optional) - multiplier factor that is applied to each
                  component in acceleration array.   
    Output:
    ------------
    desc: Description of the earthquake (e.g., name, year, etc)
    npts: total number of recorded points (acceleration data)
    dt: time interval of recorded points
    time: array (n x 1) - time array, same length with npts
    inp_acc: array (n x 1) - acceleration array, same length with time
             unit usually in (g) unless stated as other.
    '''    
    try:
        if not scalefactor:
            scalefactor = 1.0
        content = StringIO(uploaded_file.getvalue().decode("utf-8"))
        counter = 0
        desc, row4Val, acc_data = "","",[]
        for x in content:
            if counter == 1:
                desc = x
            elif counter == 3:
                row4Val = x
                if row4Val[0][0] == 'N':
                    val = row4Val.split()
                    npts = float(val[(val.index('NPTS='))+1].rstrip(','))
                    dt = float(val[(val.index('DT='))+1])
                else:
                    val = row4Val.split()
                    npts = float(val[0])
                    dt = float(val[1])
            elif counter > 3:
                data = str(x).split()
                for value in data:
                    a = float(value) * scalefactor
                    acc_data.append(a)
                inp_acc = np.asarray(acc_data)
                time = []
                for i in range (0,len(acc_data)):
                    t = i * dt
                    time.append(t)
            counter = counter + 1
        return desc, npts, dt, time, inp_acc
    except IOError:
        print("processMotion FAILED!: File is not in the directory")


def predict(model, df):    
    predictions_data = predict_model(estimator = model, data = df)
    return predictions_data['prediction_label'][0]

def find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return array[idx]
    
def SA(acc_rec, T1):
    T = np.arange(.01, 6, .01)    
    w = 2*np.pi/T 
    mass = 1 
    xi = 0.05 
    c = 2*xi*w*mass
    wd = w*np.sqrt(1-xi**2)
    p1 = -mass*acc_rec*9.81
    
    Equivalent_Velocity = np.array([])
    SA = np.zeros(len(T))
    SV = np.zeros(len(T))
    SD = np.zeros(len(T))
    
    for j in np.arange(len(T)):
        I0 = 1/w[j]**2*(1-np.exp(-xi*w[j]*dt)*(xi*w[j]/wd[j]*np.sin(wd[j]*dt)+np.cos(wd[j]*dt)))
        J0 = 1/w[j]**2*(xi*w[j]+np.exp(-xi*w[j]*dt)*(-xi*w[j]*np.cos(wd[j]*dt)+wd[j]*np.sin(wd[j]*dt)))
        AA = [[np.exp(-xi*w[j]*dt)*(np.cos(wd[j]*dt)+xi*w[j]/wd[j]*np.sin(wd[j]*dt)), np.exp(-xi*w[j]*dt)*np.sin(wd[j]*dt)/wd[j]], 
               [-w[j]**2*np.exp(-xi*w[j]*dt)*np.sin(wd[j]*dt)/wd[j], np.exp(-xi*w[j]*dt)*(np.cos(wd[j]*dt)-xi*w[j]/wd[j]*np.sin(wd[j]*dt))]]
        BB = [[I0*(1+xi/w[j]/dt)+J0/w[j]**2/dt-1/w[j]**2, -xi/w[j]/dt*I0-J0/w[j]**2/dt+1/w[j]**2 ], [J0-(xi*w[j]+1/dt)*I0, I0/dt]]
        
        u1 = np.zeros(len(acc_rec))
        udre1 = np.zeros(len(acc_rec))
        for by in range(1,len(acc_rec),1) :    
            u1[by] = AA[0][0]*u1[by-1] + AA[0][1]*udre1[by-1] + BB[0][0]*p1[by-1] + BB[0][1]*p1[by]
            udre1[by] = AA[1][0]*u1[by-1]+AA[1][1]*udre1[by-1] + BB[1][0]*p1[by-1]+BB[1][1]*p1[by]           
        udd1 = -(w[j]**2*u1+c[j]*udre1)-acc_rec*9.81
        
        SA[j] = np.max(np.abs(udd1+acc_rec*9.81))
        SV[j] = np.max(np.abs(udre1))
        SD[j] = np.max(np.abs(u1)) 

        E_I = 0
        EI =[]            
        for qr in range(len(acc_rec)):
            dE_I = -(acc_rec[qr]*udre1[qr]*9.81*dt)
            E_I = E_I + dE_I
            EI.append(E_I)        
        VE = (2*E_I)**0.5
        Equivalent_Velocity = np.append(Equivalent_Velocity, VE)  
        
    T_near_T1 = find_nearest(T, T1)
    T_num = np.where(T == T_near_T1) 
    
    SA_T1 = SA[T_num][0]
    SV_T1 = SV[T_num][0]
    SD_T1 = SD[T_num][0]
    VE_T1 = Equivalent_Velocity[T_num][0]

    Tinit = find_nearest(T, 0.1*T1)
    Tfin = find_nearest(T, 1.8*T1)
    nT = (Tfin - Tinit)/0.01 + 1
    num_init = np.where(T == Tinit)[0][0]
    num_fin = np.where(T == Tfin)[0][0] 
    
    sumSA = 0
    for dw in SA[num_init:num_fin+1]:
        sumSA = sumSA + dw
    AvSA = sumSA/nT 

    sumSV = 0
    for ev in SV[num_init:num_fin+1]:
        sumSV = sumSV + ev
    AvSV = sumSV/nT 

    sumSD = 0
    for fu in SD[num_init:num_fin+1]:
        sumSD = sumSD + fu
    AvSD = sumSD/nT 

    sumVE = 0
    for cx in Equivalent_Velocity[num_init:num_fin+1]:
        sumVE = sumVE + cx
    AvVE = sumVE/nT

    return SA_T1, SV_T1, SD_T1, VE_T1, AvSA, AvSV, AvSD, AvVE


st.markdown('***<p style="font-size:26px; color:green;">Prediction of Peak Dynamic Responses of 2D Steel Moment Frames Subjected to Earthquake Ground Motion Records</p>***', unsafe_allow_html=True)
st.sidebar.write('***<p style="color:red;">Acceleration Time History</p>***', unsafe_allow_html=True)
uploaded_file = st.sidebar.file_uploader('**Please upload an acceleration time history from the PEER NGA database.**')
if uploaded_file is not None:

    sf =st.sidebar.number_input('**Enter a scale factor for the earthquake:**')
    if sf:
        desc, npts, dt, rec_time, inp_acc = processNGAfile(uploaded_file, scalefactor=sf)
        st.write("###")
        st.write('*Description of the earthquake*: ',desc)
        st.write("*Total number of points*: ", str(int(npts)))
        st.write('*Time interval*: ',str(dt), 'sec')
        st.write('*Single-column acceleration data*:')

        d = {'Time': rec_time, 'Accel.': inp_acc}
        df_acc =  pd.DataFrame(data=d).T
        st.write(df_acc)
        st.write('*Acceleration time-history graph*:')
        fig = plt.figure(figsize=(8.3,2.3))
        plt.plot(df_acc.T.iloc[:,0], df_acc.T.iloc[:,1], linewidth=1.1, c='lightseagreen')
        plt.xticks(np.arange(0, df_acc.T.iloc[-1,0]+1, 5))
        plt.xlabel('Time (sec)', fontsize=11)
        plt.ylabel('Acceleration (g)', fontsize=11)
        plt.grid(color='lightgrey', linestyle='-', linewidth=0.4)
        st.pyplot(fig)

        check = st.checkbox('Agree')
        if check:
            st.sidebar.write("####")
            st.sidebar.write('***<p style="color:red;">Structural Properties of the Steel Frame</p>***', unsafe_allow_html=True)
            nst = st.sidebar.slider(label = 'Number of stories', min_value = 3, max_value = 13 , value = 8, step = 1)
            nbay = st.sidebar.slider(label = 'Number of bays', min_value = 2, max_value = 5 , value = 3, step = 1)
            lcol = st.sidebar.slider(label = 'Height of the stories (m)', min_value = 3.0, max_value = 4.0 , value = 3.2, step = 0.1)
            lbeam = st.sidebar.slider(label = 'Length of the bays (m)', min_value = 4.0, max_value = 7.0 , value = 6.0, step = 0.1)
            t1 = st.sidebar.number_input('Fundamental period T1 (sec)', value=1.16)
            mass = st.sidebar.number_input('Total mass (ton)', value=563.67)
            keff = st.sidebar.number_input('Effective stiffness (kN/$m^2$)', value=9407.64)
            vult = st.sidebar.number_input('Ultimate strength (kN)', value=2046.71)

            check2 = st.sidebar.checkbox('Submit')
            if check2:
                st.write("###")
                st.markdown('**<p style="font-size:20px; color:red;">OUTPUT:</p>**', unsafe_allow_html=True)
                with st.form("my_form"):
                    target_name = st.selectbox("**Select your TARGET**", ("None","Maximum Global Drift Ratio (MGDR)", "Maximum Interstory Drift Ratio (MIDR)",
                                                                          "Base Shear Coefficient (BSC)", "Maximum Floor Acceleration (MFA)")) 
                
                    submitted = st.form_submit_button("Calculate")
                    # if submitted:
                        # def choose_target():
                        #     with st.spinner(text="Operation in progress. Please wait..."):
                        #         if target_name =="Maximum Global Drift Ratio (MGDR)":
                        #             model = load_model('CatBoost_MGDR')

                        #             GM = gm.SeismoGM(dt = dt , acc = inp_acc , unit = 'g')
                        #             GM.set_units(acc="g", vel="m", disp="m")

                        #             sat1 = SA(inp_acc, t1)[0]
                        #             svt1 = SA(inp_acc, t1)[1]
                        #             sdt1 = SA(inp_acc, t1)[2]
                        #             avve = SA(inp_acc, t1)[7]
                        #             pga = GM.get_pga()
                        #             pgv = GM.get_pgv()
                        #             pgd = GM.get_pgd()
                        #             IC = GM.get_ic()

                        #             features = {'Lbay (m)':lbeam, 'Sa(T1) (g)':sat1/9.81, 'Sv(T1) (m/s)':svt1, 'Sd(T1) (m)':sdt1, 'AvVE (m/s)':avve, 
                        #                         'PGA (g)':pga, 'PGV (m/s)':pgv, 'PGD (m)':pgd, 'IC': IC, 'eff. Stiffness (kN/m^2)':keff,
                        #                         'ult. Strength (kN)': vult}
                        #             features_original = {'LBeam':lbeam, 'SA_T1':sat1, 'SV_T1':svt1, 'SD_T1':sdt1, 'AvVE':avve, 'PGA':pga, 'PGV':pgv,
                        #                                  'PGD':pgd, 'IC':IC, 'Ke_frame':keff, 'Vult_frame':vult}
                                
                        #             features_df = pd.DataFrame([features], index=['Value'])
                        #             features_df2 = pd.DataFrame([features_original], index=['Value'])
                                
                        #             st.write('**<p style="font-size:17.5px; color:#02007c;">Features:</p>**', unsafe_allow_html=True)
                        #             st.write(features_df)
                        #             prediction = predict(model, features_df2)
                        #             prediction = round(prediction,3)
                        #             st.write('**<p style="font-size:17.5px; color:#02007c;">Target:</p>**', unsafe_allow_html=True)
                        #             return annotated_text("Based on the CatBoost model, the estimated **maximum global drift ratio** of this frame is", (str(prediction)+str(' %'), "", "#8ef"))
                            
                                # if target_name =="Maximum Interstory Drift Ratio (MIDR)":
                                #     model = load_model('LightGBM_MIDR')

                                #     GM = gm.SeismoGM(dt = dt , acc = inp_acc , unit = 'g')
                                #     GM.set_units(acc="g", vel="m", disp="m")

                                #     svt1 = SA(inp_acc, t1)[1]
                                #     sdt1 = SA(inp_acc, t1)[2]
                                #     avsa = SA(inp_acc, t1)[4]
                                #     avsv = SA(inp_acc, t1)[5]
                                #     avsd = SA(inp_acc, t1)[6]
                                #     pgd = GM.get_pgd()
                                #     pgv = GM.get_pgv()
                                #     velRMS = GM.get_rms()[1]
                                #     ic = GM.get_ic()
                                #     cav = GM.get_cavdi()[0]

                                #     features = {'Nstory':nst, 'Lbay (m)':lbeam, 'T1 (s)':t1, 'Sv(T1) (m/s)':svt1, 'Sd(T1) (m)':sdt1, 'AvSa (g)':avsa/9.81,
                                #                 'AvSv (m/s)':avsv, 'AvSd (m)':avsd, 'PGV (m/s)':pgv, 'PGD (m)':pgd,'VelRMS (m/s)':velRMS, 'IC':ic,
                                #                 'CAV (m/s)':cav, 'eff. Stiffness (kN/m^2)':keff, 'ult. Strength (kN)': vult}
                                #     features_original = {'NStory':nst, 'LBeam':lbeam, 'T1':t1, 'SV_T1':svt1, 'SD_T1':sdt1, 'AvSA':avsa, 'AvSV':avsv,
                                #                          'AvSD':avsd, 'PGV':pgv, 'PGD':pgd, 'VelRMS':velRMS, 'IC':ic, 'CAV':cav, 'Ke_frame':keff,
                                #                          'Vult_frame':vult}

                                #     features_df = pd.DataFrame([features], index=['Value'])
                                #     features_df2 = pd.DataFrame([features_original], index=['Value'])
                                
                                #     st.write('**<p style="font-size:17.5px; color:#02007c;">Features:</p>**', unsafe_allow_html=True)
                                #     st.write(features_df)
                                #     prediction = predict(model, features_df2)
                                #     prediction = round(prediction,3)
                                #     st.write('**<p style="font-size:17.5px; color:#02007c;">Target:</p>**', unsafe_allow_html=True)
                                #     return annotated_text("Based on the LightGBM model, the estimated **maximum interstory drift ratio** of this frame is", (str(prediction)+str(' %'), "", "#8ef"))
                            
                                # if target_name =="Base Shear Coefficient (BSC)":
                                #     model = load_model('LightGBM_BSC')

                                #     GM = gm.SeismoGM(dt = dt , acc = inp_acc , unit = 'g')
                                #     GM.set_units(acc="g", vel="m", disp="m")

                                #     sat1 = SA(inp_acc, t1)[0]
                                #     vet1 = SA(inp_acc, t1)[3]
                                #     avsa = SA(inp_acc, t1)[4]
                                #     avsv = SA(inp_acc, t1)[5]
                                #     avsd = SA(inp_acc, t1)[6]
                                #     pga = GM.get_pga()
                                #     pgv = GM.get_pgv()
                                #     pgd = GM.get_pgd()
                                #     ic = GM.get_ic()

                                #     features = {'Lbay (m)':lbeam, 'T1 (s)':t1, 'Mass (ton)':mass ,'Sa(T1) (g)':sat1/9.81, 'VE(T1) (m/s)':vet1,
                                #                 'AvSa (g)':avsa/9.81, 'AvSv (m/s)':avsv, 'AvSd (m)':avsd, 'PGA (g)':pga, 'PGV (m/s)':pgv,
                                #                 'PGD (m)':pgd, 'IC':ic, 'eff. Stiffness (kN/m^2)':keff, 'ult. Strength (kN)':vult}
                                #     features_original = {'LBeam':lbeam, 'T1':t1, 'M':mass ,'SA_T1':sat1, 'VE_T1':vet1, 'AvSA':avsa, 'AvSV':avsv,
                                #                          'AvSD':avsd, 'PGA':pga, 'PGV':pgv, 'PGD':pgd, 'IC':ic, 'Ke_frame':keff, 'Vult_frame':vult}
                                #     features_df = pd.DataFrame([features], index=['Value'])
                                #     features_df2 = pd.DataFrame([features_original], index=['Value'])
                            
                                #     st.write('**<p style="font-size:17.5px; color:#02007c;">Features:</p>**', unsafe_allow_html=True)
                                #     st.write(features_df)
                                #     prediction = predict(model, features_df2)
                                #     prediction = round(prediction,3)
                                #     st.write('**<p style="font-size:17.5px; color:#02007c;">Target:</p>**', unsafe_allow_html=True)
                                #     return annotated_text("Based on the LightGBM model, the estimated **base shear coefficient** of this frame is", (str(prediction), "", "#8ef"))

                                # if target_name =="Maximum Floor Acceleration (MFA)":
                                #     model = load_model('LightGBM_MFA')

                                #     GM = gm.SeismoGM(dt = dt , acc = inp_acc , unit = 'g')
                                #     GM.set_units(acc="g", vel="m", disp="m")

                                #     svt1 = SA(inp_acc, t1)[1]
                                #     avsa = SA(inp_acc, t1)[4]
                                #     avsv = SA(inp_acc, t1)[5]
                                #     avsd = SA(inp_acc, t1)[6]
                                #     pgv = GM.get_pgv()
                                #     pgd = GM.get_pgd()
                                #     accRMS = GM.get_rms()[0]
                                #     ia = GM.get_ia()
                                #     Significant_Duration = GM.get_t_5_95()[0]
                                #     IF = (pgv**1.5)*((Significant_Duration)**0.5)

                                #     features = {'Lbay (m)':lbeam, 'T1 (s)':t1, 'Sv(T1) (m/s)':svt1, 'AvSa (g)':avsa/9.81, 'AvSv (m/s)':avsv,
                                #                 'AvSd (m)':avsd, 'PGV (m/s)':pgv, 'PGD (m)':pgd, 'AccRMS (g)':accRMS, 'IA (m/s)':ia, 'I PGV-Δ':IF,
                                #                 'ult. Strength (kN)':vult}
                                #     features_original = {'LBeam':lbeam, 'T1':t1, 'SV_T1':svt1, 'AvSA':avsa, 'AvSV':avsv, 'AvSD':avsd, 'PGV':pgv,
                                #                          'PGD':pgd, 'AccRMS':accRMS, 'IA':ia, 'IF':IF, 'Vult_frame':vult}
                                            
                                #     features_df = pd.DataFrame([features], index=['Value'])
                                #     features_df2 = pd.DataFrame([features_original], index=['Value'])

                                #     st.write('**<p style="font-size:17.5px; color:#02007c;">Features:</p>**', unsafe_allow_html=True)
                                #     st.write(features_df)
                                #     prediction = predict(model, features_df2)
                                #     prediction = round(prediction,3)
                                #     st.write('**<p style="font-size:17.5px; color:#02007c;">Target:</p>**', unsafe_allow_html=True)
                                #     return annotated_text("Based on the LightGBM model, the estimated **maximum floor acceleration** of this frame is", (str(prediction)+str(' g'), "", "#8ef"))

                        # choose_target()
