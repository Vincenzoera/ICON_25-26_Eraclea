import tkinter as tk
import warnings
import numpy as np
import Model as m
from tkinter import ttk, messagebox

warnings.filterwarnings('ignore')

# ================================
# FUNZIONE DI PREDIZIONE
# ================================
def predizione_prezzo():
    Label_Prediction.configure(text="")

    try:
        country = Entry_Country.get()
        index = m.model.dataframe_UI.index[m.model.dataframe_UI['country']==country].tolist()
        country = m.model.dataframe.iloc[index[0],11]

        street = Entry_Street.get()
        index = m.model.dataframe_UI.index[m.model.dataframe_UI['street']==street].tolist()
        street = m.model.dataframe.iloc[index[0],10]

        city = Entry_City.get()
        indexCitta = m.model.dataframe.columns.get_loc(f"city_{city}")

        Mq_living = float((Entry_Living.get()))
        sqft_lot = float((Entry_Lot.get()))
        sqft_basement = float((Entry_Basement.get()))
        sqft_above = float((Entry_Above.get()))
        rooms = float(Entry_Room.get())
        floors = float(Entry_Floor.get())
        waterfront = int(Entry_WF.get())  
        view = round(float((Entry_View.get())))
        condition = round(float((Entry_Cond.get())))  
        yr_built = int(Entry_YearC.get())
        yr_renovated = int(Entry_YearR.get())

        sample = np.zeros((1,57))
        sample[0,:13] = np.array([
            Mq_living,sqft_lot,floors,waterfront,view,condition,
            sqft_above,sqft_basement,yr_built,yr_renovated,street,country,rooms
        ]).reshape(1, -1)
        sample[0,indexCitta] = 1

        modelScelto = str(ComboBox_Model.get())

        if modelScelto == 'Random Forest':
            sample_scaled = scaler.transform(sample.reshape(1, -1))
            forest_modelPredict = forest_model.predict(sample_scaled)
            result_text = f"💰 Prezzo stimato: {forest_modelPredict[0]:,.2f} $"
            Label_Prediction.configure(text=result_text, fg="#D00000")

        elif modelScelto == 'SGD':
            sample_scaled = scaler_SGD.transform(sample.reshape(1, -1))
            predicted_probabilities = SGD_model.predict_proba(sample_scaled).squeeze()
            index = np.argmax(predicted_probabilities)            
            probability = predicted_probabilities[index]
            if index == len(predicted_probabilities) - 1:
                result_text = (
                    f"📊 Probabilità {probability*100:.2f}%\n"
                    f"Fascia: {price_ranges[index][0]} $ in su."
                )
            else:
                result_text = (
                    f"📊 Probabilità {probability*100:.2f}%\n"
                    f"Fascia: {price_ranges[index][0]} - {price_ranges[index][1]} $."
                )
            Label_Prediction.configure(text=result_text, fg="#D00000")

        else:
            messagebox.showerror("Errore", f"Scegli un modello valido (hai scelto {modelScelto}).")
            return

        # 🔔 Mostra anche una messagebox con il risultato
        messagebox.showinfo("Risultato Predizione", result_text)

    except Exception as e:
        messagebox.showerror("Errore", f"Controlla i campi di input!\n\nDettagli: {e}")

def update_streets(event):
    streets = m.get_Via_withCity(Entry_City.get())
    Entry_Street.config(values=streets)
    Entry_Street.current(0)
    return

# ================================
# DATI E MODELLI
# ================================
price_ranges = [
    (0, 80000),
    (80000, 150000),
    (150000, 200000),
    (200000, 650000),
    (650000, 1000000),
    (1000000, 3000000),
    (3000000, float('inf'))
]

prices_x_train, prices_x_test, prices_y_train, prices_y_test, scaler  = m.crea_basedati(modelUsed="RandomForest")
prices_x_train_SGD, prices_x_test_SGD, prices_y_train_SGD, prices_y_test_SGD, scaler_SGD  = m.crea_basedati(modelUsed="SGD")

forest_model = m.modello(prices_x_train, prices_x_test, prices_y_train, prices_y_test)
SGD_model = m.modello2(prices_x_train_SGD, prices_x_test_SGD, prices_y_train_SGD, prices_y_test_SGD)

# ================================
# GUI PRINCIPALE
# ================================
window = tk.Tk()
window.title("🏠 V_HOUSE")
window.geometry("900x900+100+50")
window.config(bg="#F5F5F5")
window.resizable(False, False)

# TITOLO
Label_Titolo = tk.Label(
    window,
    text="💡 Inserisci le informazioni sulla casa",
    font=("Segoe UI", 22, "bold"),
    bg="#F5F5F5",
    fg="#222831"
)
Label_Titolo.pack(pady=(30, 10))

# FRAME DEGLI INPUT
frame_inputs = tk.Frame(window, bg="#FFFFFF", bd=1, relief="solid", padx=50, pady=30)
frame_inputs.pack(padx=30, pady=10, fill="x")

# STILE
style = ttk.Style()
style.theme_use("clam")
style.configure("TCombobox", fieldbackground="#FFFFFF", background="#E8E8E8", padding=5)
style.configure("TSpinbox", arrowsize=14)

LABEL_FONT = ("Segoe UI", 12)
ENTRY_WIDTH = 20

def add_field(row, text, widget):
    lbl = tk.Label(frame_inputs, text=text, font=LABEL_FONT, bg="#FFFFFF", fg="#222831")
    lbl.grid(row=row, column=0, sticky="w", pady=5, padx=10)
    widget.grid(row=row, column=1, sticky="e", pady=5, padx=10)

# CAMPI
Entry_Country = ttk.Combobox(frame_inputs, values=m.get_Regione(), state='readonly', width=ENTRY_WIDTH)
Entry_Country.current(0)
add_field(0, "Nazione:", Entry_Country)

Entry_City = ttk.Combobox(frame_inputs, values=m.get_Citta(), state='readonly', width=ENTRY_WIDTH)
Entry_City.current(0)
add_field(1, "Città:", Entry_City)

Entry_Street = ttk.Combobox(frame_inputs, values=m.get_Via(), state='readonly', width=ENTRY_WIDTH)
Entry_Street.current(0)
Entry_City.bind("<<ComboboxSelected>>", update_streets)
add_field(2, "Via:", Entry_Street)

Entry_Living = ttk.Combobox(frame_inputs, values=m.get_Living(), state='normal', width=ENTRY_WIDTH)
add_field(3, "Metri quadri Vivibili:", Entry_Living)

Entry_Lot = ttk.Combobox(frame_inputs, values=m.get_Lot(), state='normal', width=ENTRY_WIDTH)
add_field(4, "Metri quadri Lotto:", Entry_Lot)

Entry_Basement = ttk.Combobox(frame_inputs, values=m.get_Basement(), state='normal', width=ENTRY_WIDTH)
add_field(5, "Metri quadri Seminterrato:", Entry_Basement)

Entry_Above = ttk.Combobox(frame_inputs, values=m.get_Above(), state='normal', width=ENTRY_WIDTH)
add_field(6, "Metri quadri Calpestabili:", Entry_Above)

Entry_YearC = ttk.Combobox(frame_inputs, values=m.get_Anno_c(), state='readonly', width=ENTRY_WIDTH)
Entry_YearC.current(0)
add_field(7, "Anno di costruzione:", Entry_YearC)

Entry_YearR = ttk.Combobox(frame_inputs, values=m.get_Anno_r(), state='readonly', width=ENTRY_WIDTH)
Entry_YearR.current(0)
add_field(8, "Anno di restauro:", Entry_YearR)

Entry_Floor = ttk.Spinbox(frame_inputs, from_=0, to=4, increment=0.5, width=ENTRY_WIDTH)
Entry_Floor.set(1.0)
add_field(9, "Piani:", Entry_Floor)

Entry_WF = ttk.Spinbox(frame_inputs, from_=0, to=1, increment=1, width=ENTRY_WIDTH)
Entry_WF.set(0)
add_field(10, "Affaccio sul mare:", Entry_WF)

Entry_Room = ttk.Spinbox(frame_inputs, from_=1, to=10, increment=1, width=ENTRY_WIDTH)
Entry_Room.set(2)
add_field(11, "Stanze:", Entry_Room)

Entry_View = ttk.Spinbox(frame_inputs, from_=0, to=5, increment=1, width=ENTRY_WIDTH)
Entry_View.set(0)
add_field(12, "Vista:", Entry_View)

Entry_Cond = ttk.Spinbox(frame_inputs, from_=1, to=5, increment=1, width=ENTRY_WIDTH)
Entry_Cond.set(3)
add_field(13, "Condizione:", Entry_Cond)

ComboBox_Model = ttk.Combobox(frame_inputs, values=m.get_Modello(), state='readonly', width=ENTRY_WIDTH)
ComboBox_Model.current(0)
add_field(14, "Modello di predizione:", ComboBox_Model)

# ================================
# PULSANTE PREDIZIONE
# ================================
getValue_button = tk.Button(
    window,
    text="🔮 Avvia Predizione",
    font=("Segoe UI", 14, "bold"),
    bg="#0F52BA",
    fg="white",
    activebackground="#1E90FF",
    activeforeground="white",
    relief="raised",
    padx=20, pady=10,
    command=predizione_prezzo
)
getValue_button.pack(pady=(20, 10))

# ================================
# LABEL DI OUTPUT
# ================================
Label_Prediction = tk.Label(
    window,
    text="",
    fg="#D00000",
    font=("Segoe UI", 16, "bold"),
    bg="#F5F5F5",
    justify="center"
)
Label_Prediction.pack(pady=15)

# ================================
# LOOP PRINCIPALE
# ================================
if __name__ == "__main__":
    window.mainloop()
