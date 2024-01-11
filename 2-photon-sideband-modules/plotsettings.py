import matplotlib as mpl


""

figsize_half = (10, 6)
figsize_full = (22, 6)

# Give colours in hex
qubit     = '#664277'  # qubit
cavity1   = '#fa476f'  # cavity 1
cavity2   = '#c86464'  # cavity 2
coupling1 = '#68b799'  # coupling between qubit and cavity 1
coupling2 = '#7eb6a1'  # coupling between qubit and cavity 1
drive     = '#68b799'  # single-tone drive strength
driveq    = 'green'    # qubit-friendly drive tone strength
drivec    = 'cyan'     # cavity-friendly drive tone strenth
sbred     = '#c86464'  # P(e0)-P(g1), red sideband transitions
sbblue    = '#6196b3'  # P(e1)-P(g0), blue sideband transitions
hline     = 'gray'     # horizontal domain lines
colormap  = 'gist_heat'   # colormap for color plots

# (line, marker, marker size)
larger       = ('-', '+', 200)
smaller      = ('-.', 'x', 120)
analytical   = ('-.', 'x', 120)
simulated    = ('--', '+', 200)
experimental = ('--', '.', 100)
Duffing = ('-', '+', 200)
Kerr = ('-.', 'x', 120)

alpha = 1  # opacity of lines

mpl.rcParams['font.size'] = 22
mpl.rcParams['axes.titlesize'] = 22
mpl.rcParams['axes.labelsize'] = 20
mpl.rcParams['xtick.labelsize'] = 18
mpl.rcParams['ytick.labelsize'] = 18
mpl.rcParams['legend.fontsize'] = 17
mpl.rcParams['figure.titlesize'] = 22
mpl.rcParams['lines.linewidth'] = 2


""

plotcolours = {
    'qubit' : qubit,
    'cavity1' : cavity1,
    'cavity2' : cavity2,
    'cavity' : cavity1,
    'coupling1' : coupling1,
    'coupling2' : coupling2,
    'coupling' : coupling1,
    'drive' : drive,
    'driveq' : driveq,
    'drivec' : drivec,
    'sbred' : sbred,
    'sbblue' : sbblue,
    'hline' : hline,
    'colormap' : colormap
}

linestyles = {
    'larger' : larger[0],
    'smaller' : smaller[0],
    'analytical' : analytical[0],
    'simulated' : simulated[0],
    'experimental' : experimental[0],
    'Duffing' : Duffing[0],
    'Kerr' : Kerr[0]
}

markerstyles = {
    'larger' : larger[1],
    'smaller' : smaller[1],
    'analytical' : analytical[1],
    'simulated' : simulated[1],
    'experimental' : experimental[1],
    'Duffing' : Duffing[1],
    'Kerr' : Kerr[1]
}

markersizes = {
    'larger' : larger[2],
    'smaller' : smaller[2],
    'analytical' : analytical[2],
    'simulated' : simulated[2],
    'experimental' : experimental[2],
    'Duffing' : Duffing[2],
    'Kerr' : Kerr[2]
}

mpl.rcParams['figure.figsize'] = figsize_half
