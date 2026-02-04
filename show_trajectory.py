"""Quick display of the probability cone"""
import json

with open('results/trajectory_chart_data.json') as f:
    data = json.load(f)

print('=' * 60)
print('CRUDE OIL - 10-DAY PROBABILITY CONE')
print('=' * 60)

dates = data['dates']
forecast = data['series']['forecast']
upper90 = data['series']['conf_90_upper']
lower90 = data['series']['conf_90_lower']

print()
print('     Date     | Forecast |   90% Range   |  Uncertainty')
print('-' * 60)

for i, date in enumerate(dates):
    spread = upper90[i] - lower90[i]
    bar_len = int(spread * 15)
    bar = '#' * bar_len if bar_len > 0 else ''
    print(f' {date} |  ${forecast[i]:6.2f}  | ${lower90[i]:.2f}-${upper90[i]:.2f} | {bar}')

print('-' * 60)
print()

# Calculate key metrics
current = forecast[0]
final = forecast[-1]
move = final - current
move_pct = move / current * 100

max_upside = max(upper90) - current
max_downside = current - min(lower90)

print(f'Current:      ${current:.2f}')
print(f'D+10 Target:  ${final:.2f} ({move:+.2f}, {move_pct:+.1f}%)')
print(f'Max Upside:   ${max(upper90):.2f} (+${max_upside:.2f})')  
print(f'Max Downside: ${min(lower90):.2f} (-${max_downside:.2f})')

print()
print('The cone WIDENS over time showing increasing uncertainty.')
print('Narrower cone at D+1 = higher confidence.')
print('Wider cone at D+10 = more uncertainty.')
