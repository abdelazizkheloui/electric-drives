import numpy as np
import math
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# Fonction pour tracer un vecteur à partir de ses coordonnées polaires
def plot_vector(x0, y0, magnitude, angle_deg, color, label, shift_x, shift_y):
    angle_rad = np.deg2rad(angle_deg)  # Conversion de l'angle en radians
    x = magnitude * np.cos(angle_rad)  # Projection sur l'axe x
    y = magnitude * np.sin(angle_rad)  # Projection sur l'axe y
    # Tracer le vecteur (flèche)
    plt.quiver(x0, y0, x, y, angles='xy', scale_units='xy', scale=1, color=color, label=label)
    plt.text((x0+x+shift_x) * 1.05, (y0+y+shift_y) * 1.05, label, fontsize=10)
    return x,y

def cartesian_to_polar(x, y):
    r = math.sqrt(x**2 + y**2)  # Calcul de la distance r (norme du vecteur)
    theta = math.atan2(y, x)    # Calcul de l'angle theta en radians (angle du vecteur)
    return r, theta

def polar_to_cartesian(magnitude, angle_rad):
    x = magnitude * np.cos(angle_rad)
    y = magnitude * np.sin(angle_rad)
    return x, y

def duplicate_elements(lst, n):
    # Utiliser une compréhension de liste pour dupliquer chaque élément n fois
    return [elem for elem in lst for _ in range(n)]

def sign(x):
    if x > 0:
        return 1
    elif x < 0:
        return -1
    else:
        return 0

def create_fmm_slot(interval, magnitude, shift):
    y = np.zeros(len(interval))
    for i, x in enumerate(interval):
        x = x + shift
        if x < 0: x = x + 2*np.pi
        if x > 2*np.pi : x = x % 2*np.pi
        if x == 0 : y[i] = -magnitude
        if x > 0 and x < np.pi : y[i] = magnitude
        if x >= np.pi and x < 2*np.pi : y[i] = -magnitude
        if x == 2*np.pi : y[i] = magnitude
    return y

radius_stator_ext = 6
radius_stator_int = 4
airgap = 0.2
radius_rotor = radius_stator_int - airgap
N_slots = 36
p = 1
m = 3
delta = 2*np.pi/N_slots
alpha = 2*np.pi/m
pitch = np.pi/p
q = int(N_slots/(2*p*m))
sequence = ['-a','b','-c','a','-b','c']
current = ['-', '-', '+', '+', '+', '-']
color = ['red','blue','green','red','blue','green']
color_bis = ['blue','blue','red','red','red','blue']
winding_sequence = duplicate_elements(sequence, q)
winding_current = duplicate_elements(current, q)
winding_color = duplicate_elements(color, q)
winding_color_bis = duplicate_elements(color_bis, q)
N_shift = q // 2    # quotient

slot_width = 0.4
slot_height = 0.8
x0 = -slot_width/2
y0 = radius_stator_int
r0, theta_0 = cartesian_to_polar(x0, y0)
x_w0 = 0
y_w0 = y0 + slot_height + 0.5
r_w0, theta_w0 = cartesian_to_polar(x_w0, y_w0)
x_c0 = 0
y_c0 = y0 + slot_height/2
r_c0, theta_c0 = cartesian_to_polar(x_c0, y_c0)


# Création d'un axe
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
fig.canvas.manager.set_window_title("Stator total mmf repartition in the airgap over time")

# Configuration des limites des axes
    
ax1.set_xlim(-9, 9)
ax1.set_ylim(-9, 9)
ax1.set_aspect('equal')
ax1.axis(False)
ax1.grid(False)
ax1.text(0, -8.5, 'Ampere-turn distribution in stator slots over time', ha='center', fontweight='bold', fontsize=8)
point_1 = plt.Circle((-8.5, -6), slot_width/4, edgecolor='red', facecolor='red')
point_2 = plt.Circle((-8.5, -7), slot_width/4, edgecolor='blue', facecolor='blue')
ax1.add_patch(point_1)
ax1.add_patch(point_2)
ax1.text(-8, -6.15, 'positive current', fontsize='6', ha='left')
ax1.text(-8, -7.15, 'negative current', fontsize='6', ha='left')

ax2.text(np.pi, 3*q+1, 'q = '+ str(q) +' slots/pole/phase', ha='center', fontsize=12, fontweight='bold')
ax2.xaxis.set_ticks([0, np.pi, 2 * np.pi], [r"$-\pi$", "0", r"$\pi$"])
ax2.set_ylim(-3*q, 3*q)
ax2.set_xlabel(r"$\theta$")
ax2.set_ylabel('Magnetomotive force')
ax2.grid()

circle_stator_ext = plt.Circle((0, 0), radius_stator_ext, edgecolor='black', facecolor='lightgray')
circle_stator_int = plt.Circle((0, 0), radius_stator_int, edgecolor='black', facecolor='white')
circle_rotor = plt.Circle((0, 0), radius_rotor, edgecolor='black', facecolor='lightgray')

ax1.add_patch(circle_stator_ext)
ax1.add_patch(circle_stator_int)
ax1.add_patch(circle_rotor)

ax1.plot([0, 0], [-5,5], 'k--', zorder = 1)
ax1.plot([-5,5], [0, 0], 'k--', zorder = 1)

rect = []
for i in range(N_slots):
    if q % 2 == 0: shift = delta/2 - (i+1)*delta
    if q % 2 == 1: shift = delta - (i+1)*delta
    x, y = polar_to_cartesian(r0, theta_0 + shift)
    rect.append(plt.Rectangle((x, y), slot_width, slot_height, angle = shift*180/np.pi, edgecolor='black', facecolor='white'))

for slot in rect:
    ax1.add_patch(slot)


for i in range(N_slots):
    if q % 2 == 0: shift = N_shift*delta + delta/2 - (i+1)*delta
    if q % 2 == 1: shift = N_shift*delta + delta - (i+1)*delta
    xw, yw = polar_to_cartesian(r_w0, theta_w0 + shift)
    ax1.text(xw, yw, winding_sequence[i], fontsize = 8, ha='center', va='center', color=winding_color[i], weight='bold')

text = []
current_circles = []
for i in range(N_slots):
    if q % 2 == 0: shift = N_shift*delta + delta/2 - (i+1)*delta
    if q % 2 == 1: shift = N_shift*delta + delta - (i+1)*delta
    xc, yc = polar_to_cartesian(r_c0, theta_c0 + shift)
    text.append(ax1.text(xc, yc, '', fontsize = 8, ha='center', va='center', color=winding_color[i], weight='bold'))
    current_circles.append(plt.Circle((xc, yc), slot_width/2, edgecolor=winding_color_bis[i], facecolor=winding_color_bis[i]))


for circle in current_circles:
    ax1.add_patch(circle)

ax1.quiver(0, 0, 7, 0, angles='xy', scale_units='xy', scale=1, color='red', label='A')
ax1.text(6.8, 0.2, 'A axis', fontsize=10)
x, y = polar_to_cartesian(7, alpha)
ax1.quiver(0, 0, x, y, angles='xy', scale_units='xy', scale=1, color='blue', label='B')
ax1.text(x - 0.4, y, 'B axis', fontsize=10)
x, y = polar_to_cartesian(7, 2*alpha)
ax1.quiver(0, 0, x, y, angles='xy', scale_units='xy', scale=1, color='green', label='C')
ax1.text(x+0.4, y-0.4, 'C axis', fontsize=10)

currents_sequence = ['Im', '-Im/2', '-Im/2']
vecteur = ax1.quiver(0, 0, 0, 0, angles='uv', scale_units='xy', scale=1, color='magenta')
vecteur_label = ax1.text(0, 0, ' ', fontsize=12, fontweight='bold', ha='right')
ia_text = ax1.text(-1, 8.5, ' ', fontsize = 8, ha='left')
ib_text = ax1.text(-1, 7.5, ' ', fontsize = 8, ha='left')
ic_text = ax1.text(-1, 6.5, ' ', fontsize = 8, ha='left')
omega_text = ax1.text(-5.5, 7.5, ' ', fontsize = 8, ha='left')


angle = np.linspace(0, 2*np.pi, 1000)
fmm_slot_a = np.zeros((q, len(angle)))
fmm_slot_b = np.zeros((q, len(angle)))
fmm_slot_c = np.zeros((q, len(angle)))

line_fmm_total, = ax2.plot([], [], 'r-', lw=2)
line_fmm_fundamental, = ax2.plot([], [], 'm--', lw=1)



def update(frame):
    amp_turn_a = np.cos((delta)*frame)
    amp_turn_b = np.cos((delta)*frame - alpha)
    amp_turn_c = np.cos((delta)*frame - 2*alpha)

    omega_text.set_text(r"$\omega_s t = $" + str(round(frame*delta*180/np.pi % 360, 1)) + '°')
    ia_text.set_text(r"$N_si_a =$" + str(round(amp_turn_a, 2)) + ' p.u')
    ib_text.set_text(r"$N_si_b =$" + str(round(amp_turn_b, 2)) + ' p.u')
    ic_text.set_text(r"$N_si_c =$" + str(round(amp_turn_c, 2)) + ' p.u')

    for i in range(N_slots):
        if q % 2 == 0: shift = N_shift*delta + delta/2 - (i+1)*delta
        if q % 2 == 1: shift = N_shift*delta + delta - (i+1)*delta
        xc, yc = polar_to_cartesian(r_c0, theta_c0 + shift + (delta) * frame)
        # text[i].set_position((xc, yc))
        # text[i].set_text(winding_current[i])
        # text[i].set_color(winding_color_bis[i])
        sequence_value = winding_sequence[i]
        if sequence_value == 'a' or sequence_value == '-a' : amp_value = amp_turn_a
        if sequence_value == 'b' or sequence_value == '-b' : amp_value = amp_turn_b
        if sequence_value == 'c' or sequence_value == '-c' : amp_value = amp_turn_c
        if amp_value > 0:
            if sequence_value[0] == '-': color_value = 'blue'
            if sequence_value[0] != '-': color_value = 'red'
        elif amp_value < 0:
            if sequence_value[0] == '-': color_value = 'red'
            if sequence_value[0] != '-': color_value = 'blue'
        current_circles[i].set_radius(slot_width * np.abs(amp_value)/2)
        current_circles[i].set_facecolor(color_value)
        current_circles[i].set_edgecolor(color_value)


    for k in range(q):
        fmm_slot_a[k] = create_fmm_slot(angle, amp_turn_a, -k*delta - delta/2 - np.pi/3)
        fmm_slot_b[k] = create_fmm_slot(angle, amp_turn_b, -k*delta - alpha - delta/2 - np.pi/3)
        fmm_slot_c[k] = create_fmm_slot(angle, amp_turn_c, -k*delta - 2*alpha - delta/2 - np.pi/3)

    fmm_phase_a = np.zeros(len(angle))
    fmm_phase_b = np.zeros(len(angle))
    fmm_phase_c = np.zeros(len(angle))
    for i in range(q):
        fmm_phase_a = fmm_phase_a + fmm_slot_a[i]
        fmm_phase_b = fmm_phase_b + fmm_slot_b[i]
        fmm_phase_c = fmm_phase_c + fmm_slot_c[i]

    fmm_total = fmm_phase_a + fmm_phase_b + fmm_phase_c
    line_fmm_total.set_data(angle, fmm_total)

    spectrum = np.fft.fft(fmm_total)
    frequencies = np.fft.fftfreq(len(spectrum), 1/(2*np.pi))
    spectrum_filtered = np.zeros_like(spectrum)
    index_fundamental = np.argmax(np.abs(spectrum[:len(spectrum)//2]))
    spectrum_filtered[index_fundamental] = spectrum[index_fundamental]
    spectrum_filtered[-index_fundamental] = spectrum[-index_fundamental]
    fmm_fundamental = np.fft.ifft(spectrum_filtered)
    line_fmm_fundamental.set_data(angle, fmm_fundamental)

    x_vect, y_vect = polar_to_cartesian(np.abs(max(fmm_fundamental))*0.5, delta * frame)
    x_label, y_label = polar_to_cartesian(2.5, (delta) * frame + 10*np.pi/180)
    vecteur.set_UVC(x_vect, y_vect)
    vecteur_label.set_position((x_label, y_label))
    vecteur_label.set_text(r"$\mathscr{F}_s$")

    return text + [vecteur] + [vecteur_label] + [ia_text] + [ib_text] + [ic_text] + [omega_text] + [line_fmm_total] + current_circles + [line_fmm_fundamental]

ani = FuncAnimation(fig, update, frames=int(2*np.pi/delta), interval=1000, blit=True, repeat=True)

ani.save("Slots_Fmm_Animation.gif", writer="pillow", fps=2)

plt.show()