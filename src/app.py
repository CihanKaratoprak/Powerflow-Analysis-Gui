import tkinter as tk
from tkinter import messagebox, ttk, filedialog
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import networkx as nx
import json
from datetime import datetime

DEVELOPER_ID = "{7F3C9D77-448A-AFF4-CKPOWERFLOW-2025}"
DEVELOPER_NAME = "Cihan KARATOPRAK"
DEVELOPER_LINK = "https://www.linkedin.com/in/cihankaratoprak"
DEVELOPER_EMAIL = "cihankarato@example.com"  # gerçek mailini koyabilirsin


# --- GLOBAL SABİTLER ---
BASE_MVA = 100.0      # Baz güç [MVA]
MAX_ITER = 20         # Maksimum iterasyon sayısı
TOL = 1e-6            # Konverjans toleransı

# Türkiye'de yaygın kullanılan bazı ACSR iletkenlerin örnek kütüphanesi
# Değerler örnektir, istersen TEİAŞ/TEDAŞ tablolarına göre revize edebilirsin.
STANDART_ILETKENLER = {
    "ACSR Sparrow 50": {"r_ohm_km": 0.641, "x_ohm_km": 0.386, "b_uS_km": 1.5},
    "ACSR Pigeon 100": {"r_ohm_km": 0.320, "x_ohm_km": 0.380, "b_uS_km": 2.0},
    "ACSR Rabbit 100": {"r_ohm_km": 0.320, "x_ohm_km": 0.384, "b_uS_km": 2.0},
    "ACSR Lynx 240": {"r_ohm_km": 0.125, "x_ohm_km": 0.400, "b_uS_km": 3.0},
    "ACSR Mink 150": {"r_ohm_km": 0.206, "x_ohm_km": 0.390, "b_uS_km": 2.5},
    "ACSR Wolf 170": {"r_ohm_km": 0.184, "x_ohm_km": 0.395, "b_uS_km": 2.7},
    "ACSR Moose 520": {"r_ohm_km": 0.068, "x_ohm_km": 0.360, "b_uS_km": 4.5},
    "Custom": {"r_ohm_km": None, "x_ohm_km": None, "b_uS_km": None},
}

# --- GLOBAL VERİ YAPILARI ---
bus_data = []
line_data = []
bildirim_var = None

# Son hesaplamadan kalan veriler (istersen kullanırsın, kısa devre artık bunlara bağlı değil)
last_bus_df = None
last_line_df = None
last_V = None
last_delta = None
last_Y = None
last_line_flows_df = None
last_cikti_klasor = None


# --- YARDIMCI FONKSİYONLAR ---

def masaustu_klasor():
    desktop = os.path.join(os.path.expanduser("~"), "Desktop")
    output_dir = os.path.join(desktop, "GucAkisiCiktisi")
    os.makedirs(output_dir, exist_ok=True)
    return output_dir


def kaydet_matris(matris, dosya_adi):
    df = pd.DataFrame(matris)
    df.to_csv(os.path.join(masaustu_klasor(), dosya_adi), index=False, header=False)


def bus_tipleri(df: pd.DataFrame) -> pd.DataFrame:
    """Gerilim ve açı bilgisine göre Slack / PV / PQ sınıflandırması yapar."""
    types = []
    for _, row in df.iterrows():
        if not pd.isna(row["Gerilim (V)"]) and not pd.isna(row["Açı (°)"]):
            types.append("Slack")
        elif not pd.isna(row["Gerilim (V)"]):
            types.append("PV")
        else:
            types.append("PQ")
    df["Tip"] = types
    return df


def ybus_olustur(n, df: pd.DataFrame, baz_gucu=BASE_MVA):
    """Hat verilerinden Ybus matrisi oluşturur (basit p.u. yaklaşımı)."""
    Y = np.zeros((n, n), dtype=complex)
    for _, row in df.iterrows():
        i = int(row['Bağlantı Başlangıcı']) - 1
        j = int(row['Bağlantı Sonu']) - 1
        r = row['Direnç (R)']
        x = row['Reaktans (X)']
        b = row['B/2']
        if r == 0 and x == 0:
            continue
        # Basit p.u. yaklaşımı (örnek amaçlı)
        z_pu = complex(r, x) * (1.0 / baz_gucu)
        y_line = 1 / z_pu
        b_total = complex(0, b)
        Y[i, i] += y_line + b_total
        Y[j, j] += y_line + b_total
        Y[i, j] -= y_line
        Y[j, i] -= y_line
    return Y


def devre_semasi_ciz(bus_data_local, line_data_local, bus_types, cikti_klasor):
    """NetworkX ile bara-hat devre şeması çizer."""
    G = nx.Graph()
    bus_labels = {}

    for idx, row in enumerate(bus_data_local):
        barano = int(row[0])
        label = f"Bara {barano}\n"
        label += f"{bus_types[idx]} / V={row[1] if row[1] else '-'} / θ={row[2] if row[2] else '-'}\n"
        label += f"Pyük={row[3]}, Qyük={row[4]}\n"
        label += f"Püretim={row[5]}"
        bus_labels[barano] = label
        G.add_node(barano)

    for row in line_data_local:
        f, t, r, x, b = row[0], row[1], row[2], row[3], row[4]
        label = f"R={r}\nX={x}\nB/2={b}"
        G.add_edge(f, t, label=label)

    pos = nx.spring_layout(G, seed=44)
    plt.figure(figsize=(9, 7))
    nx.draw(
        G, pos, with_labels=True,
        node_size=2200, node_color="#dff9fb",
        font_size=9, font_weight='bold', edge_color="#30336b"
    )
    nx.draw_networkx_labels(G, pos, labels=bus_labels, font_size=7)
    edge_labels = nx.get_edge_attributes(G, 'label')
    nx.draw_networkx_edge_labels(
        G, pos, edge_labels=edge_labels, font_size=8,
        bbox=dict(boxstyle='round,pad=0.2', fc='white', ec='black', lw=0.5)
    )
    plt.title("Devre Şeması (Bara & Hatlar)", fontsize=14, fontweight='bold')
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(os.path.join(cikti_klasor, "devre_semasi.png"))
    plt.close()


def tek_hat_semasi_ciz(bus_df: pd.DataFrame, cikti_klasor: str):
    """
    Profesyonel tek hat diyagramı:
    - Baralar x ekseninde hizalanır
    - Her baranın üstünde: Bara No, V, açı
    - Altında: yük ve üretim (P, Q)
    - Gerilim seviyesine göre renk kodlama
    """
    if bus_df.empty:
        return

    sirali_bus = bus_df.sort_values("Barano")
    x_positions = {int(row["Barano"]): i for i, (_, row) in enumerate(sirali_bus.iterrows())}

    plt.figure(figsize=(13, 4))

    xs = [x_positions[int(row["Barano"])] for _, row in sirali_bus.iterrows()]
    ys = [0 for _ in xs]

    plt.plot(
        xs, ys,
        linestyle='-', linewidth=2.5, color="#34495e",
        marker='o', markersize=10,
        markerfacecolor="#3498db", markeredgecolor="black"
    )

    for _, row in sirali_bus.iterrows():
        barano = int(row["Barano"])
        x = x_positions[barano]

        V = row["Gerilim (V)"]
        delta = row["Açı (°)"]
        P_yuk = row["P_yük"]
        Q_yuk = row["Q_yük"]
        P_uretim = row["P_üretim"]
        Q_uretim = row["Q_üretim"]

        v_text = "-" if pd.isna(V) else f"{V:.3f} p.u."
        d_text = "-" if pd.isna(delta) else f"{delta:.2f}°"
        if pd.isna(Q_uretim):
            q_gen_text = "-"
        else:
            q_gen_text = f"{Q_uretim:.2f}"

        color_v = "#2c3e50"
        if not pd.isna(V):
            if V < 0.95:
                color_v = "#e74c3c"  # düşük gerilim
            elif V > 1.05:
                color_v = "#f1c40f"  # yüksek gerilim
            else:
                color_v = "#27ae60"  # normal

        plt.text(
            x, 0.55,
            f"Bara {barano}\nV={v_text},  δ={d_text}",
            ha="center", va="bottom",
            fontsize=10, fontweight="bold", color=color_v
        )

        plt.text(
            x - 0.2, -0.75,
            f"Yük\nP={P_yuk:.2f}\nQ={Q_yuk:.2f}",
            ha="right", va="top",
            fontsize=9, color="#e74c3c", fontweight="bold"
        )

        if not pd.isna(P_uretim) and (P_uretim != 0 or (not pd.isna(Q_uretim) and Q_uretim != 0)):
            plt.text(
                x + 0.2, -0.75,
                f"Üretim\nP={P_uretim:.2f}\nQ={q_gen_text}",
                ha="left", va="top",
                fontsize=9, color="#27ae60", fontweight="bold"
            )

    plt.ylim(-1.2, 1.0)
    plt.xlim(min(xs) - 0.5, max(xs) + 0.5)

    plt.title("Tek Hat Üzerinde Yük ve Üretim Konumları", fontsize=13, fontweight="bold")
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(os.path.join(cikti_klasor, "tek_hat_yuk_konumlari.png"))
    plt.close()


def hat_guc_akislari_hesapla(bus_df: pd.DataFrame, V, delta, line_df: pd.DataFrame):
    """
    Hat başı/sonu güç akışlarını, kayıpları ve yüzde yüklenmeyi hesaplar.
    Basit p.u. modeline göre yaklaşık hesaplama.
    """
    V_complex = V * np.exp(1j * delta)
    records = []

    for _, row in line_df.iterrows():
        i = int(row["Bağlantı Başlangıcı"]) - 1
        j = int(row["Bağlantı Sonu"]) - 1
        r = row["Direnç (R)"]
        x = row["Reaktans (X)"]
        b = row["B/2"]

        if r == 0 and x == 0:
            continue

        z_pu = complex(r, x) * (1.0 / BASE_MVA)
        y_series = 1 / z_pu
        y_shunt = 1j * b

        Vi = V_complex[i]
        Vj = V_complex[j]

        I_ij = (Vi - Vj) * y_series + Vi * y_shunt
        I_ji = (Vj - Vi) * y_series + Vj * y_shunt

        S_ij = Vi * np.conj(I_ij) * BASE_MVA
        S_ji = Vj * np.conj(I_ji) * BASE_MVA

        P_ij, Q_ij = S_ij.real, S_ij.imag
        P_ji, Q_ji = S_ji.real, S_ji.imag

        P_loss = P_ij + P_ji
        Q_loss = Q_ij + Q_ji

        S_mag = np.abs(S_ij)
        percent_loading = 100.0 * S_mag / BASE_MVA if BASE_MVA > 0 else 0.0

        records.append({
            "From": i + 1,
            "To": j + 1,
            "P_from_MW": P_ij,
            "Q_from_Mvar": Q_ij,
            "P_to_MW": P_ji,
            "Q_to_Mvar": Q_ji,
            "P_loss_MW": P_loss,
            "Q_loss_Mvar": Q_loss,
            "S_from_MVA": S_mag,
            "Loading_%": percent_loading
        })

    if not records:
        return pd.DataFrame()

    return pd.DataFrame(records)


# --- ÇÖZÜM YÖNTEMLERİ ---

def guc_akisi_nr(bus_df: pd.DataFrame, Y: np.ndarray):
    """
    Basit NR-benzeri iteratif çözüm.
    (Tam Newton-Raphson yerine eğitim amaçlı bir iteratif yaklaşım.)
    """
    n = len(bus_df)
    G, B = Y.real, Y.imag

    V = np.ones(n)
    delta = np.zeros(n)

    for i, row in bus_df.iterrows():
        if not pd.isna(row["Gerilim (V)"]):
            V[i] = row["Gerilim (V)"]
        if not pd.isna(row["Açı (°)"]):
            delta[i] = np.radians(row["Açı (°)"])

    iterasyonlar = []
    loglar = []

    for iter_no in range(1, MAX_ITER + 1):
        P = np.zeros(n)
        Q = np.zeros(n)

        for i in range(n):
            for j in range(n):
                angle = delta[i] - delta[j]
                P[i] += V[i] * V[j] * (G[i, j] * np.cos(angle) + B[i, j] * np.sin(angle))
                Q[i] += V[i] * V[j] * (G[i, j] * np.sin(angle) - B[i, j] * np.cos(angle))

        dP, dQ = [], []
        for i in range(n):
            tip = bus_df.loc[i, "Tip"]
            P_net = bus_df.loc[i, "P_üretim"] - bus_df.loc[i, "P_yük"]
            Q_net = (bus_df.loc[i, "Q_üretim"] - bus_df.loc[i, "Q_yük"]
                     if not pd.isna(bus_df.loc[i, "Q_üretim"]) else 0.0)

            if tip in ["PQ", "PV"]:
                dP.append(P_net - P[i])
            if tip == "PQ":
                dQ.append(Q_net - Q[i])

        mismatch_vector = np.array(dP + dQ)
        max_hata = np.linalg.norm(mismatch_vector, np.inf)

        iterasyonlar.append({
            "iterasyon": iter_no,
            **{f"V{i+1}": V[i] for i in range(n)},
            **{f"delta{i+1}(deg)": np.degrees(delta[i]) for i in range(n)},
            "max_hata": max_hata
        })

        loglar.append(
            f"İterasyon {iter_no}: max hata={max_hata:.6e}\n"
            f"V: {np.round(V, 6)}\n"
            f"Delta (deg): {np.round(np.degrees(delta), 6)}\n"
        )

        if max_hata < TOL:
            loglar.append(f"Konverjans {iter_no}. adımda sağlandı.\n")
            break

        pq_idx = [i for i in range(n) if bus_df.loc[i, "Tip"] == "PQ"]
        if pq_idx:
            V[pq_idx] += 0.01 * np.sign(V[pq_idx])
            delta[pq_idx] += np.radians(0.5)

    return V, delta, iterasyonlar, loglar


def guc_akisi_dc(bus_df: pd.DataFrame, Y: np.ndarray):
    """
    Basit DC Load Flow yaklaşımı.
    Gerilim genlikleri 1 p.u. kabul edilir, yalnızca açı çözümlenir.
    """
    n = len(bus_df)
    B = Y.imag

    slack_idx = 0
    for i in range(n):
        if bus_df.loc[i, "Tip"] == "Slack":
            slack_idx = i
            break

    P_net = np.array(bus_df["P_üretim"] - bus_df["P_yük"], dtype=float)

    non_slack = [i for i in range(n) if i != slack_idx]
    Bp = B[np.ix_(non_slack, non_slack)]

    try:
        delta_ns = np.linalg.solve(Bp, -P_net[non_slack])
    except np.linalg.LinAlgError:
        delta_ns = np.zeros(len(non_slack))

    delta = np.zeros(n)
    for idx, i in enumerate(non_slack):
        delta[i] = delta_ns[idx]

    V = np.ones(n)

    iterasyonlar = [{
        "iterasyon": 1,
        **{f"V{i+1}": V[i] for i in range(n)},
        **{f"delta{i+1}(deg)": np.degrees(delta[i]) for i in range(n)},
        "max_hata": 0.0
    }]
    loglar = [
        "DC Load Flow tek adımda çözüldü.\n"
        f"Delta (deg): {np.round(np.degrees(delta), 6)}\n"
    ]

    return V, delta, iterasyonlar, loglar


# --- TABLO GÜNCELLEME ---

def guncelle_bus_tablosu():
    for i in bus_tree.get_children():
        bus_tree.delete(i)
    for idx, row in enumerate(bus_data):
        bus_tree.insert("", "end", values=row,
                        tags=('oddrow',) if idx % 2 else ('evenrow',))


def guncelle_hat_tablosu():
    for i in hat_tree.get_children():
        hat_tree.delete(i)
    for idx, row in enumerate(line_data):
        hat_tree.insert("", "end", values=row,
                        tags=('oddrow',) if idx % 2 else ('evenrow',))


# --- SENARYO KAYDET / YÜKLE ---

def senaryoyu_kaydet():
    """
    Mevcut bara ve hat verilerini,
    masaüstündeki 'GucAkisiCiktisi' klasörüne
    otomatik isimli bir JSON senaryo dosyası olarak kaydeder.
    """
    if not bus_data or not line_data:
        messagebox.showerror(
            "Hata",
            "Kaydedilecek senaryo için önce bara ve hat verilerini giriniz."
        )
        return

    cikti_klasor = masaustu_klasor()
    zaman = datetime.now().strftime("%Y%m%d_%H%M%S")
    dosya_adi = f"senaryo_{zaman}.json"
    path = os.path.join(cikti_klasor, dosya_adi)

    data = {
        "bus_data": bus_data,
        "line_data": line_data
    }

    try:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

        messagebox.showinfo(
            "Bilgi",
            f"Senaryo masaüstündeki 'GucAkisiCiktisi' klasörüne\n"
            f"'{dosya_adi}' adıyla kaydedildi."
        )
    except Exception as e:
        messagebox.showerror("Hata", f"Senaryo kaydedilemedi: {e}")


def senaryoyu_yukle():
    global bus_data, line_data
    path = filedialog.askopenfilename(
        filetypes=[("JSON Dosyası", "*.json")],
        title="Senaryo Yükle"
    )
    if not path:
        return
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        bus_data = data.get("bus_data", [])
        line_data = data.get("line_data", [])
        guncelle_bus_tablosu()
        guncelle_hat_tablosu()
        messagebox.showinfo("Bilgi", "Senaryo başarıyla yüklendi.")
    except Exception as e:
        messagebox.showerror("Hata", f"Senaryo yüklenemedi: {e}")


# --- KISA DEVRE ANALİZİ ---

def kisa_devre_analizi():
    """
    Basit 3-faz kısa devre analizi (p.u.):
    - bus_data ve line_data'dan Ybus oluşturur
    - Zbus = Ybus^-1 (pseudo-inverse) hesaplar
    - Her bara için Thevenin empedansı ve 3-faz Ik (p.u.) üretir
    """
    if not bus_data or not line_data:
        messagebox.showerror(
            "Hata",
            "Kısa devre analizi için önce bara ve hat verilerini giriniz."
        )
        return

    bus_df = pd.DataFrame(bus_data, columns=[
        "Barano", "Gerilim (V)", "Açı (°)",
        "P_yük", "Q_yük", "P_üretim", "Q_üretim", "Q_min", "Q_max"
    ])
    line_df = pd.DataFrame(line_data, columns=[
        "Bağlantı Başlangıcı", "Bağlantı Sonu",
        "Direnç (R)", "Reaktans (X)", "B/2",
        "Uzunluk (km)", "İletken"
    ])

    bus_df = bus_tipleri(bus_df)
    n = len(bus_df)

    Y = ybus_olustur(n, line_df, baz_gucu=BASE_MVA)

    try:
        Zbus = np.linalg.pinv(Y)
    except Exception as e:
        messagebox.showerror("Hata", f"Zbus hesaplanamadı: {e}")
        return

    records = []
    for k in range(n):
        Zkk = Zbus[k, k]
        Z_mag = np.abs(Zkk)
        Ik_pu = 0.0 if Z_mag == 0 else 1.0 / Z_mag

        records.append({
            "Bara": int(bus_df.loc[k, "Barano"]),
            "Z_th_pu": Z_mag,
            "Ik3faz_pu": Ik_pu
        })

    df_sc = pd.DataFrame(records)

    sc_win = tk.Toplevel(pencere)
    sc_win.title("3-Faz Kısa Devre Analizi (p.u.)")

    tree = ttk.Treeview(
        sc_win,
        columns=["Bara", "Z_th_pu", "Ik3faz_pu"],
        show="headings"
    )
    for col in ["Bara", "Z_th_pu", "Ik3faz_pu"]:
        tree.heading(col, text=col)
        tree.column(col, anchor="center", width=120)
    tree.pack(fill="both", expand=True, padx=10, pady=10)

    for _, row in df_sc.iterrows():
        tree.insert(
            "",
            "end",
            values=(
                int(row["Bara"]),
                round(row["Z_th_pu"], 6),
                round(row["Ik3faz_pu"], 3)
            )
        )

    cikti_klasor = masaustu_klasor()
    dosya_adi = "kisa_devre_3faz_pu.xlsx"
    df_sc.to_excel(os.path.join(cikti_klasor, dosya_adi), index=False)

    messagebox.showinfo(
        "Bilgi",
        f"3-faz kısa devre sonuçları pencerede gösterildi ve\n"
        f"masaüstündeki 'GucAkisiCiktisi' klasörüne\n"
        f"'{dosya_adi}' adıyla kaydedildi."
    )


# --- BUTON FONKSİYONLARI ---

def ekle_bus():
    try:
        barano = int(barano_entry.get())
        gerilim = float(gerilim_entry.get()) if gerilim_entry.get() else None
        aci = float(aci_entry.get()) if aci_entry.get() else None
        p_yuk = float(p_yuk_entry.get())
        q_yuk = float(q_yuk_entry.get())
        p_uretim = float(p_uretim_entry.get())
        q_uretim = float(q_uretim_entry.get()) if q_uretim_entry.get() else None
        q_min = float(q_min_entry.get()) if q_min_entry.get() else None
        q_max = float(q_max_entry.get()) if q_max_entry.get() else None
    except ValueError as e:
        messagebox.showerror("Hata", f"Veri hatası: {e}")
        return

    bus_data.append([barano, gerilim, aci, p_yuk, q_yuk,
                     p_uretim, q_uretim, q_min, q_max])
    guncelle_bus_tablosu()
    for entry in [barano_entry, gerilim_entry, aci_entry,
                  p_yuk_entry, q_yuk_entry, p_uretim_entry,
                  q_uretim_entry, q_min_entry, q_max_entry]:
        entry.delete(0, tk.END)


def hesapla_hat_parametreleri():
    """Seçilen iletken ve uzunluğa göre R, X, B/2 değerlerini otomatik hesaplar."""
    try:
        uzunluk_km = float(uzunluk_entry.get())
    except ValueError:
        messagebox.showerror("Hata", "Hat uzunluğunu (km) sayısal giriniz.")
        return

    iletken_adi = iletken_combo.get()
    data = STANDART_ILETKENLER.get(iletken_adi)

    if data is None:
        messagebox.showerror("Hata", "Geçersiz iletken tipi.")
        return

    if iletken_adi == "Custom":
        return

    r = data["r_ohm_km"] * uzunluk_km
    x = data["x_ohm_km"] * uzunluk_km
    b_toplam = data["b_uS_km"] * uzunluk_km

    r_entry.delete(0, tk.END)
    x_entry.delete(0, tk.END)
    b_entry.delete(0, tk.END)

    r_entry.insert(0, f"{r:.5f}")
    x_entry.insert(0, f"{x:.5f}")
    b_entry.insert(0, f"{b_toplam:.5f}")


def ekle_hat():
    try:
        f = int(from_entry.get())
        t = int(to_entry.get())
        r = float(r_entry.get())
        x = float(x_entry.get())
        b = float(b_entry.get())
        uzunluk_km = float(uzunluk_entry.get()) if uzunluk_entry.get() else 0.0
        iletken_adi = iletken_combo.get() if iletken_combo.get() else "Custom"

        line_data.append([f, t, r, x, b, uzunluk_km, iletken_adi])
        guncelle_hat_tablosu()

        for entry in [from_entry, to_entry, r_entry, x_entry, b_entry, uzunluk_entry]:
            entry.delete(0, tk.END)
        iletken_combo.set("")
    except Exception as e:
        messagebox.showerror("Hata", f"Veri hatası: {e}")


def hesapla():
    global bildirim_var, last_bus_df, last_line_df, last_V, last_delta, last_Y, last_line_flows_df, last_cikti_klasor

    if not bus_data or not line_data:
        messagebox.showerror("Eksik Veri", "Baralar veya hat verileri eksik.")
        return

    bus_df = pd.DataFrame(bus_data, columns=[
        "Barano", "Gerilim (V)", "Açı (°)",
        "P_yük", "Q_yük", "P_üretim", "Q_üretim", "Q_min", "Q_max"
    ])
    line_df = pd.DataFrame(line_data, columns=[
        "Bağlantı Başlangıcı", "Bağlantı Sonu",
        "Direnç (R)", "Reaktans (X)", "B/2",
        "Uzunluk (km)", "İletken"
    ])

    bus_df = bus_tipleri(bus_df)
    n = len(bus_df)

    Y = ybus_olustur(n, line_df, baz_gucu=BASE_MVA)
    kaydet_matris(Y, "Ybus.csv")

    yontem = cozum_yontemi_var.get()
    if "DC Load Flow" in yontem:
        V, delta, iterasyonlar, loglar = guc_akisi_dc(bus_df, Y)
    else:
        V, delta, iterasyonlar, loglar = guc_akisi_nr(bus_df, Y)

    result_win = tk.Toplevel(pencere)
    result_win.title("Güç Akışı Sonuçları")

    frame_res = ttk.Frame(result_win)
    frame_res.pack(fill="both", expand=True, padx=10, pady=10)

    tree = ttk.Treeview(
        frame_res,
        columns=["Barano", "Tip", "Gerilim (p.u.)", "Açı (°)"],
        show="headings"
    )
    for col in ["Barano", "Tip", "Gerilim (p.u.)", "Açı (°)"]:
        tree.heading(col, text=col)
        tree.column(col, anchor="center", width=140)

    res_scroll_y = ttk.Scrollbar(frame_res, orient="vertical", command=tree.yview)
    res_scroll_x = ttk.Scrollbar(frame_res, orient="horizontal", command=tree.xview)

    tree.configure(yscrollcommand=res_scroll_y.set, xscrollcommand=res_scroll_x.set)

    tree.grid(row=0, column=0, sticky="nsew")
    res_scroll_y.grid(row=0, column=1, sticky="ns")
    res_scroll_x.grid(row=1, column=0, sticky="ew")

    frame_res.rowconfigure(0, weight=1)
    frame_res.columnconfigure(0, weight=1)

    for i in range(n):
        tree.insert(
            "",
            "end",
            values=(
                int(bus_df.loc[i, "Barano"]),
                bus_df.loc[i, "Tip"],
                round(V[i], 4),
                round(np.degrees(delta[i]), 4)
            )
        )

    cikti_klasor = masaustu_klasor()

    sonuc_df = pd.DataFrame({
        "Barano": [int(bus_df.loc[i, "Barano"]) for i in range(n)],
        "Tip": bus_df["Tip"].tolist(),
        "Gerilim (p.u.)": V.round(4),
        "Açı (°)": np.degrees(delta).round(4)
    })
    sonuc_df.to_excel(os.path.join(cikti_klasor, "sonuc.xlsx"), index=False)

    iterasyon_df = pd.DataFrame(iterasyonlar)
    iterasyon_df.to_csv(os.path.join(cikti_klasor, "iterasyonlar.csv"), index=False)

    with open(os.path.join(cikti_klasor, "log.txt"), "w", encoding="utf-8") as f:
        for satir in loglar:
            f.write(satir + "\n")

    devre_semasi_ciz(bus_data, line_data, bus_df["Tip"].tolist(), cikti_klasor)

    plt.figure()
    plt.plot(sonuc_df["Barano"], sonuc_df["Gerilim (p.u.)"], marker='o')
    plt.title("Gerilim Profili", fontsize=14, fontweight="bold")
    plt.xlabel("Bara No", fontsize=12)
    plt.ylabel("Gerilim (p.u.)", fontsize=12)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(cikti_klasor, "gerilim_profili.png"))
    plt.close()

    tek_hat_semasi_ciz(bus_df, cikti_klasor)

    line_flows_df = hat_guc_akislari_hesapla(bus_df, V, delta, line_df)
    if not line_flows_df.empty:
        line_flows_df.to_excel(os.path.join(cikti_klasor, "hat_guc_akislari.xlsx"), index=False)

    last_bus_df = bus_df
    last_line_df = line_df
    last_V = V
    last_delta = delta
    last_Y = Y
    last_line_flows_df = line_flows_df
    last_cikti_klasor = cikti_klasor

    bildirim_text = (
        "Hesaplama tamamlandı.\n"
        "Tüm çıktılar ve grafikler masaüstündeki 'GucAkisiCiktisi' klasörüne kaydedildi."
    )
    messagebox.showinfo("Başarılı", bildirim_text)
    if bildirim_var is not None:
        bildirim_var.config(text=bildirim_text)


# --- ANA ARAYÜZ ---

pencere = tk.Tk()
pencere.title("Güç Akışı Hesaplayıcı")
pencere.state('zoomed')
pencere.configure(bg="#1e272e")
pencere.iconbitmap("app_icon.ico")


style = ttk.Style()
try:
    style.theme_use("clam")
except tk.TclError:
    pass

style.configure("Main.TFrame", background="#1e272e")
style.configure("Card.TLabelframe", background="#2f3640", foreground="#f5f6fa", borderwidth=0)
style.configure("Card.TLabelframe.Label", background="#2f3640", foreground="#f5f6fa", font=("Segoe UI", 10, "bold"))
style.configure("Header.TLabel", background="#273c75", foreground="#f5f6fa", font=("Segoe UI", 18, "bold"))
style.configure("SubHeader.TLabel", background="#273c75", foreground="#dcdde1", font=("Segoe UI", 10, "italic"))
style.configure("TLabel", background="#2f3640", foreground="#f5f6fa", font=("Segoe UI", 9))
style.configure("Accent.TButton", font=("Segoe UI", 10, "bold"), padding=6)
style.map("Accent.TButton",
          background=[("!disabled", "#00a8ff"), ("pressed", "#0097e6"), ("active", "#0097e6")],
          foreground=[("!disabled", "#f5f6fa")])
style.configure("Primary.TButton", font=("Segoe UI", 12, "bold"), padding=8)
style.map("Primary.TButton",
          background=[("!disabled", "#e84118"), ("pressed", "#c23616"), ("active", "#c23616")],
          foreground=[("!disabled", "#f5f6fa")])

style.configure("Treeview",
                background="#353b48",
                foreground="#f5f6fa",
                fieldbackground="#353b48",
                rowheight=24,
                font=("Segoe UI", 9))
style.configure("Treeview.Heading",
                background="#2f3640",
                foreground="#f5f6fa",
                font=("Segoe UI", 9, "bold"))

# --- ANA SCROLLABLE ÇERÇEVE ---
outer_frame = ttk.Frame(pencere, style="Main.TFrame")
outer_frame.pack(fill="both", expand=True)

canvas = tk.Canvas(
    outer_frame,
    bg="#1e272e",
    highlightthickness=0
)
canvas.grid(row=0, column=0, sticky="nsew")

scroll_y = ttk.Scrollbar(outer_frame, orient="vertical", command=canvas.yview)
scroll_y.grid(row=0, column=1, sticky="ns")

scroll_x = ttk.Scrollbar(outer_frame, orient="horizontal", command=canvas.xview)
scroll_x.grid(row=1, column=0, sticky="ew")

canvas.configure(yscrollcommand=scroll_y.set, xscrollcommand=scroll_x.set)

scroll_frame = ttk.Frame(canvas, style="Main.TFrame")
canvas_window = canvas.create_window((0, 0), window=scroll_frame, anchor="nw")


def on_frame_configure(event):
    canvas.configure(scrollregion=canvas.bbox("all"))


def on_canvas_configure(event):
    canvas.itemconfig(canvas_window, width=event.width)


scroll_frame.bind("<Configure>", on_frame_configure)
canvas.bind("<Configure>", on_canvas_configure)

outer_frame.rowconfigure(0, weight=1)
outer_frame.columnconfigure(0, weight=1)

# --- HEADER ---
header_frame = ttk.Frame(scroll_frame, style="Main.TFrame")
header_frame.pack(fill="x")

header_bar = tk.Frame(header_frame, bg="#273c75", height=60)
header_bar.pack(fill="x")

header_label = ttk.Label(
    header_bar,
    text="GÜÇ AKIŞI HESAPLAYICI",
    style="Header.TLabel"
)
header_label.pack(side="left", padx=20, pady=10)

subheader_label = ttk.Label(
    header_bar,
    text="Power Flow Analysis Tool • Cihan KARATOPRAK",
    style="SubHeader.TLabel"
)
subheader_label.pack(side="left", padx=10, pady=10)

# --- MAIN FRAME (sol / sağ) ---
main_frame = ttk.Frame(scroll_frame, style="Main.TFrame", padding=15)
main_frame.pack(fill="both", expand=True)

left_frame = ttk.Frame(main_frame, style="Main.TFrame")
left_frame.pack(side="left", fill="both", expand=True, padx=(0, 8))

right_frame = ttk.Frame(main_frame, style="Main.TFrame")
right_frame.pack(side="right", fill="both", expand=True, padx=(8, 0))

# Çözüm yöntemi seçimi
method_frame = ttk.Frame(left_frame, style="Main.TFrame")
method_frame.pack(fill="x", pady=(0, 10))

cozum_yontemi_var = tk.StringVar(value="NR (mevcut)")

yontem_label = ttk.Label(method_frame, text="Çözüm Yöntemi:", font=("Segoe UI", 10, "bold"))
yontem_label.pack(side="left", padx=(0, 5))

yontem_combo = ttk.Combobox(
    method_frame,
    textvariable=cozum_yontemi_var,
    values=["NR (mevcut)", "DC Load Flow (hızlı yaklaşık)"],
    width=30,
    state="readonly"
)
yontem_combo.pack(side="left")

# Bara veri girişi
frm_bus = ttk.Labelframe(
    left_frame, text="BARA VERİ GİRİŞİ",
    style="Card.TLabelframe", padding=10
)
frm_bus.pack(fill="both", expand=True, pady=(0, 10))

labels1 = ["Barano", "Gerilim (V)", "Açı (°)", "P_yük", "Q_yük"]
for idx, text in enumerate(labels1):
    ttk.Label(frm_bus, text=text).grid(row=0, column=idx, padx=4, pady=2, sticky="w")

barano_entry = ttk.Entry(frm_bus, width=7)
gerilim_entry = ttk.Entry(frm_bus, width=7)
aci_entry = ttk.Entry(frm_bus, width=7)
p_yuk_entry = ttk.Entry(frm_bus, width=7)
q_yuk_entry = ttk.Entry(frm_bus, width=7)
entry_list1 = [barano_entry, gerilim_entry, aci_entry, p_yuk_entry, q_yuk_entry]
for idx, entry in enumerate(entry_list1):
    entry.grid(row=1, column=idx, padx=4, pady=2)

labels2 = ["P_üretim", "Q_üretim", "Q_min", "Q_max"]
for idx, text in enumerate(labels2):
    ttk.Label(frm_bus, text=text).grid(row=2, column=idx, padx=4, pady=2, sticky="w")
p_uretim_entry = ttk.Entry(frm_bus, width=7)
q_uretim_entry = ttk.Entry(frm_bus, width=7)
q_min_entry = ttk.Entry(frm_bus, width=7)
q_max_entry = ttk.Entry(frm_bus, width=7)
entry_list2 = [p_uretim_entry, q_uretim_entry, q_min_entry, q_max_entry]
for idx, entry in enumerate(entry_list2):
    entry.grid(row=3, column=idx, padx=4, pady=2)

ttk.Button(
    frm_bus, text="Bara Ekle", command=ekle_bus,
    style="Accent.TButton"
).grid(row=3, column=4, padx=8, pady=2, sticky="e")

bus_tree = ttk.Treeview(
    frm_bus,
    columns=labels1 + labels2,
    show="headings", height=5
)
for col in labels1 + labels2:
    bus_tree.heading(col, text=col)
    bus_tree.column(col, width=80, anchor="center")
bus_tree.grid(row=4, column=0, columnspan=10, pady=8, sticky="nsew")

frm_bus.rowconfigure(4, weight=1)
for c in range(10):
    frm_bus.columnconfigure(c, weight=1)

# Hat veri girişi
frm_hat = ttk.Labelframe(
    right_frame, text="HAT VERİ GİRİŞİ",
    style="Card.TLabelframe", padding=10
)
frm_hat.pack(fill="both", expand=True)

ttk.Label(frm_hat, text="Bağlantı Başlangıcı").grid(row=0, column=0, padx=4, pady=2, sticky="w")
ttk.Label(frm_hat, text="Bağlantı Sonu").grid(row=0, column=1, padx=4, pady=2, sticky="w")
ttk.Label(frm_hat, text="Hat Uzunluğu (km)").grid(row=0, column=2, padx=4, pady=2, sticky="w")
ttk.Label(frm_hat, text="İletken Tipi").grid(row=0, column=3, padx=4, pady=2, sticky="w")

from_entry = ttk.Entry(frm_hat, width=7)
to_entry = ttk.Entry(frm_hat, width=7)
uzunluk_entry = ttk.Entry(frm_hat, width=10)

from_entry.grid(row=1, column=0, padx=4, pady=2)
to_entry.grid(row=1, column=1, padx=4, pady=2)
uzunluk_entry.grid(row=1, column=2, padx=4, pady=2)

iletken_combo = ttk.Combobox(
    frm_hat,
    values=list(STANDART_ILETKENLER.keys()),
    width=18, state="readonly"
)
iletken_combo.grid(row=1, column=3, padx=4, pady=2)

ttk.Button(
    frm_hat, text="Parametre Hesapla", command=hesapla_hat_parametreleri,
    style="Accent.TButton"
).grid(row=1, column=4, padx=8, pady=2)

ttk.Label(frm_hat, text="Direnç R (p.u.)").grid(row=2, column=0, padx=4, pady=2, sticky="w")
ttk.Label(frm_hat, text="Reaktans X (p.u.)").grid(row=2, column=1, padx=4, pady=2, sticky="w")
ttk.Label(frm_hat, text="B/2 (p.u.)").grid(row=2, column=2, padx=4, pady=2, sticky="w")

r_entry = ttk.Entry(frm_hat, width=10)
x_entry = ttk.Entry(frm_hat, width=10)
b_entry = ttk.Entry(frm_hat, width=10)

r_entry.grid(row=3, column=0, padx=4, pady=2)
x_entry.grid(row=3, column=1, padx=4, pady=2)
b_entry.grid(row=3, column=2, padx=4, pady=2)

ttk.Button(
    frm_hat, text="Hat Ekle", command=ekle_hat,
    style="Accent.TButton"
).grid(row=3, column=3, columnspan=2, padx=8, pady=4, sticky="w")

hat_tree = ttk.Treeview(
    frm_hat,
    columns=[
        "Bağlantı Başlangıcı", "Bağlantı Sonu",
        "Direnç (R)", "Reaktans (X)", "B/2",
        "Uzunluk (km)", "İletken"
    ],
    show="headings", height=5
)
for col in [
    "Bağlantı Başlangıcı", "Bağlantı Sonu",
    "Direnç (R)", "Reaktans (X)", "B/2",
    "Uzunluk (km)", "İletken"
]:
    hat_tree.heading(col, text=col)
    hat_tree.column(col, anchor="center", width=90)

hat_tree.grid(row=4, column=0, columnspan=5, pady=8, sticky="nsew")

frm_hat.rowconfigure(4, weight=1)
for c in range(5):
    frm_hat.columnconfigure(c, weight=1)

# Alt çerçeve
bottom_frame = ttk.Frame(scroll_frame, style="Main.TFrame", padding=(15, 0, 15, 15))
bottom_frame.pack(fill="x")

scenario_frame = ttk.Frame(bottom_frame, style="Main.TFrame")
scenario_frame.pack(side="left", padx=(0, 15))

ttk.Button(
    scenario_frame,
    text="Senaryoyu Kaydet",
    command=senaryoyu_kaydet,
    style="Accent.TButton"
).pack(side="left", padx=5)

ttk.Button(
    scenario_frame,
    text="Senaryoyu Yükle",
    command=senaryoyu_yukle,
    style="Accent.TButton"
).pack(side="left", padx=5)

ttk.Button(
    scenario_frame,
    text="Kısa Devre Analizi",
    command=kisa_devre_analizi,
    style="Accent.TButton"
).pack(side="left", padx=5)

bildirim_var = ttk.Label(
    bottom_frame, text="",
    font=("Segoe UI", 10),
    foreground="#2ecc71",
    background="#1e272e"
)
bildirim_var.pack(side="left")

ttk.Button(
    bottom_frame,
    text="GÜÇ AKIŞI HESAPLA",
    command=hesapla,
    style="Primary.TButton"
).pack(side="right")

pencere.mainloop()
