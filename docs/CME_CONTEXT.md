# Nexus — CME Partnership Context

## Relationship
Nexus is being developed in context of a partnership/engagement with the **CME Group** (Chicago Mercantile Exchange). Key stakeholders include hedging desks and procurement teams.

## What This Means for Product

### Audience Priority (Reordered)
1. **Hedging desks** — Primary. Commodity hedgers, corporate treasury, risk managers
2. **Institutional/Pro traders** — Futures, options, spreads
3. **Wealth managers** — Institutional allocators
4. **Hedge funds** — Systematic and discretionary
5. **Retail traders** — Secondary, but important for scale story
6. **Casual investors** — Tertiary

### CME-Specific Considerations
- **Asset classes**: Futures, options, commodities (ag, energy, metals), interest rates, FX, equity index
- **Hedging language**: Basis risk, hedge ratios, correlation, exposure, Greeks, VaR
- **Procurement requirements**: Enterprise security, SOC 2, data governance, SLAs
- **Integration potential**: CME Globex data feeds, CME DataMine, CME Smart Stream

### Procurement Conversations
Procurement teams will want to know:
- Where does the model data come from?
- How are models validated?
- What's the governance/audit trail?
- Uptime SLAs?
- Data retention policies?
- Can we white-label?

### Design Implications
- Must look **institutional**, not startup-y
- Clean, dense, Bloomberg-adjacent aesthetic
- Every signal needs provenance (which models, what inputs, confidence intervals)
- PDF/report export for client-facing wealth managers
- API-first for institutional integration
- Dark mode default (trading standard)

## Key Terminology
When talking to CME people, use:
- "Model ensemble consensus" not "AI predictions"
- "Risk signals" not "buy/sell signals"  
- "Factor attribution" not "why it thinks this"
- "Backtested accuracy" not "it's usually right"
- "Governance framework" not "we track stuff"

## Timeline
- **Feb 2**: Artemis (Farzaneh) joins for design collaboration
- **TBD**: CME stakeholder conversations begin

---

*This doc evolves as we learn more from CME conversations.*
