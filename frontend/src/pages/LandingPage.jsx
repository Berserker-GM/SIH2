export default function LandingPage({ onEnter }) {
  return (
    <div className="nn-home">
      {/* HEADER */}
      <header className="nn-header">
        <button className="nn-wordmark" onClick={() => window.scrollTo(0, 0)}>
          NORMNATIVE
        </button>

        <div className="nn-header-meta">
          <span className="nn-online-dot" />
          WORLD MODEL ONLINE
          <span className="nn-meta-separator" />
          HORIZON 30 SEC
        </div>
      </header>

      {/* HERO */}
      <main className="nn-hero">
        <section className="nn-intro">
          <div className="nn-kicker">
            NETWORK WORLD MODEL / PREDICTIVE CYBER DEFENCE
          </div>

          <h1>
            Predict what the
            <br />
            network does <span>next.</span>
          </h1>

          <p>
            NORMNATIVE observes the recent state of a network,
            understands its present structure, and forecasts how
            that state may evolve over the next 30 seconds.
          </p>

          <div className="nn-actions">
            <button className="nn-enter" onClick={onEnter}>
              ENTER COMMAND
              <span>→</span>
            </button>

            <span className="nn-build">WORLD MODEL / BUILD 0.1</span>
          </div>
        </section>

        {/* MODEL DIAGRAM */}
        <section className="nn-model">
          <div className="nn-model-header">
            <span>01 / WORLD STATE</span>
            <span className="nn-model-status">LIVE MODEL</span>
          </div>

          <div className="nn-model-stage">

            {/* TOP AXIS */}
            <div className="nn-axis">

              <div className="nn-state nn-state-past">
                <div className="nn-state-number">−40</div>
                <div className="nn-state-title">PAST</div>
                <div className="nn-state-value">8 STATES</div>
                <div className="nn-state-note">40 SEC OBSERVED</div>
              </div>

              <div className="nn-flow-line nn-line-left">
                <span />
                <span />
                <span />
                <span />
              </div>

              <div className="nn-state nn-state-now">
                <div className="nn-now-ring">
                  <div className="nn-now-core" />
                </div>

                <div className="nn-state-title">NOW</div>
                <div className="nn-state-value">S_t / G_t</div>
                <div className="nn-state-note">CURRENT WORLD STATE</div>
              </div>

              <div className="nn-flow-line nn-line-right future">
                <span />
                <span />
                <span />
                <span />
                <span />
                <span />
              </div>

              <div className="nn-state nn-state-future">
                <div className="nn-state-number">+30</div>
                <div className="nn-state-title">FUTURE</div>
                <div className="nn-state-value">
                  Ŝt+1 → Ŝt+6
                </div>
                <div className="nn-state-note">30 SEC FORECAST</div>
              </div>

            </div>

            {/* CENTER VERTICAL FLOW */}
            <div className="nn-analysis-flow">
              <div className="nn-drop-line">
                <span className="nn-drop-node" />
              </div>

              <div className="nn-analysis-block">
                <span>WHAT / WHY</span>
                <strong>INTERPRET THE FORECAST</strong>
              </div>

              <div className="nn-vertical-connector" />

              <div className="nn-output-grid">
                <div>
                  <span>WHAT</span>
                  <strong>MITRE</strong>
                  <small>ATTACK PROGRESSION</small>
                </div>

                <div>
                  <span>WHY</span>
                  <strong>FEATURES</strong>
                  <small>MODEL EVIDENCE</small>
                </div>
              </div>

              <div className="nn-vertical-connector short" />

              <button className="nn-investigate" onClick={onEnter}>
                <span>INVESTIGATION</span>
                <strong>TRACE THE DECISION</strong>
                <b>→</b>
              </button>
            </div>
          </div>

          {/* TECHNICAL FOOTER */}
          <div className="nn-model-footer">
            <div>
              <span>INPUT</span>
              <strong>CIC-IDS2018</strong>
            </div>

            <div>
              <span>STATE</span>
              <strong>32-D VECTOR</strong>
            </div>

            <div>
              <span>WINDOW</span>
              <strong>5 SEC</strong>
            </div>

            <div>
              <span>CONTEXT</span>
              <strong>8 STATES</strong>
            </div>

            <div>
              <span>ROLLOUT</span>
              <strong>6 STEPS</strong>
            </div>
          </div>
        </section>
      </main>

      {/* BOTTOM STRIP */}
      <footer className="nn-footer">
        <span>NORMNATIVE / NETWORK WORLD MODEL</span>

        <div>
          <span>PAST 40S</span>
          <span>NOW</span>
          <span>FUTURE 30S</span>
        </div>
      </footer>
    </div>
  );
}