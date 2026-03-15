"""
utils/alerting.py — APEX ALGO Email Alert System

Email-only alerting via smtplib (Python stdlib). Every alert is an HTML
email with inline CSS, APEX ALGO branding, and deduplication.

Design rules (Decision 7):
  • Every public method is wrapped in try/except — alerting NEVER crashes trading.
  • Every attempt is logged to alerts_log via MemoryStore.
  • Dedup by (subject + body[:100]) with priority-based cooldowns.
  • Daily cap: 50 emails max.
"""

import hashlib
import logging
import smtplib
import time
from datetime import datetime, timezone, date
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from typing import Optional

import config

logger = logging.getLogger(__name__)

# ═════════════════════════════════════════════════════════════════════
# BRAND IDENTITY
# ═════════════════════════════════════════════════════════════════════

BRAND = {
    "name": "APEX ALGO",
    "tagline": "Autonomous. Intelligent. Disciplined.",
    "primary": "#0D1B3E",
    "accent": "#2563EB",
    "success": "#059669",
    "warning": "#D97706",
    "danger": "#DC2626",
    "intelligence": "#7C3AED",
    "neutral": "#374151",
    "background": "#F3F4F6",
    "surface": "#FFFFFF",
}

# Priority → (X-Priority header, cooldown seconds)
PRIORITY_MAP = {
    "CRITICAL": ("1", 300),      # 5 minutes
    "HIGH":     ("2", 900),      # 15 minutes
    "NORMAL":   (None, 3600),    # 1 hour
}

DAILY_EMAIL_LIMIT = 50


# ═════════════════════════════════════════════════════════════════════
# ALERT MANAGER
# ═════════════════════════════════════════════════════════════════════

class AlertManager:
    """APEX ALGO email alert hub. 7 canonical templates, zero SMS."""

    def __init__(self, cfg=config, memory_store=None):
        self.cfg = cfg
        self._memory_store = memory_store

        # SMTP readiness
        self._enabled = bool(
            getattr(cfg, "SMTP_HOST", "") and
            getattr(cfg, "SMTP_USER", "") and
            getattr(cfg, "SMTP_PASS", "") and
            getattr(cfg, "ALERT_EMAIL", "")
        )

        if self._enabled:
            logger.info("APEX ALGO email alerts enabled → %s",
                        cfg.ALERT_EMAIL)
        else:
            logger.info("SMTP not configured — email alerts disabled.")

        # Dedup: hash → last-sent timestamp
        self._dedup_cache: dict[str, float] = {}
        # Daily counter: date_str → count
        self._daily_count: dict[str, int] = {}

        # Dashboard URL for CTA buttons
        self._dashboard_url = (
            f"http://{getattr(cfg, 'DASHBOARD_HOST', 'localhost')}:"
            f"{getattr(cfg, 'DASHBOARD_PORT', 8501)}"
        )

    # ─────────────────────────────────────────────────────────────────
    # PRIVATE — Email Transport
    # ─────────────────────────────────────────────────────────────────

    def _send_email(self, subject: str, html_body: str,
                    priority: str = "NORMAL") -> bool:
        """
        Send branded HTML email with dedup + daily limit.

        Returns True if sent, False if suppressed/failed.
        """
        # ── Dedup check ──
        dedup_key = hashlib.sha256(
            (subject + html_body[:100]).encode()
        ).hexdigest()
        now = time.time()
        _, cooldown = PRIORITY_MAP.get(priority, (None, 3600))
        last_sent = self._dedup_cache.get(dedup_key)

        if last_sent and (now - last_sent) < cooldown:
            logger.debug("Email suppressed (dedup %ss): %s", cooldown,
                         subject[:60])
            self._log_alert(subject, "suppressed", priority=priority)
            return False

        # ── Daily limit ──
        today_str = date.today().isoformat()
        today_count = self._daily_count.get(today_str, 0)

        if today_count >= DAILY_EMAIL_LIMIT:
            if today_count == DAILY_EMAIL_LIMIT:
                # Send one final warning, then stop
                self._daily_count[today_str] = today_count + 1
                logger.warning("Daily email limit (%d) reached",
                               DAILY_EMAIL_LIMIT)
            self._log_alert(subject, "suppressed", error="daily_limit",
                            priority=priority)
            return False

        if not self._enabled:
            logger.warning("Email (no SMTP): %s", subject[:80])
            self._log_alert(subject, "failed", error="smtp_not_configured",
                            priority=priority)
            return False

        try:
            msg = MIMEMultipart("alternative")
            msg["Subject"] = subject
            msg["From"] = f"APEX ALGO <{self.cfg.SMTP_USER}>"
            msg["To"] = self.cfg.ALERT_EMAIL

            cc = getattr(self.cfg, "ALERT_EMAIL_CC", "")
            if cc:
                msg["Cc"] = cc

            x_priority, _ = PRIORITY_MAP.get(priority, (None, 3600))
            if x_priority:
                msg["X-Priority"] = x_priority

            msg.attach(MIMEText(html_body, "html"))

            recipients = [self.cfg.ALERT_EMAIL]
            if cc:
                recipients.append(cc)

            port = int(getattr(self.cfg, "SMTP_PORT", 587))
            with smtplib.SMTP(self.cfg.SMTP_HOST, port, timeout=15) as srv:
                srv.ehlo()
                srv.starttls()
                srv.login(self.cfg.SMTP_USER, self.cfg.SMTP_PASS)
                srv.sendmail(self.cfg.SMTP_USER, recipients,
                             msg.as_string())

            self._dedup_cache[dedup_key] = now
            self._daily_count[today_str] = today_count + 1
            logger.info("Email sent → %s", subject[:80])
            self._log_alert(subject, "sent", priority=priority)
            return True

        except Exception as exc:
            logger.error("Email send failed: %s", exc)
            self._log_alert(subject, "failed", error=str(exc),
                            priority=priority)
            return False

    def _log_alert(self, subject: str, status: str,
                   error: str = None, priority: str = "NORMAL") -> None:
        """Log email attempt to MemoryStore (best-effort)."""
        if self._memory_store is None:
            return
        try:
            self._memory_store.log_alert({
                "alert_type": priority,
                "subject": subject,
                "recipient": getattr(self.cfg, "ALERT_EMAIL", ""),
                "status": status,
                "error": error or "",
            })
        except Exception:
            pass  # Never crash on logging

    # ─────────────────────────────────────────────────────────────────
    # PRIVATE — HTML Templates
    # ─────────────────────────────────────────────────────────────────

    @staticmethod
    def _data_row(label: str, value: str,
                  value_color: str = "#111827") -> str:
        """Single key/value row for email data tables."""
        return (
            f'<tr>'
            f'<td style="padding:10px 16px;font-family:Arial,Helvetica,'
            f'sans-serif;font-size:14px;color:#6B7280;border-bottom:'
            f'1px solid #E5E7EB;width:40%;">{label}</td>'
            f'<td style="padding:10px 16px;font-family:Arial,Helvetica,'
            f'sans-serif;font-size:14px;color:{value_color};font-weight:'
            f'600;border-bottom:1px solid #E5E7EB;">{value}</td>'
            f'</tr>'
        )

    def _base_template(
        self,
        header_color: str,
        icon: str,
        title: str,
        subtitle: str,
        rows: str,
        notice: str = "",
        footer_note: str = "",
    ) -> str:
        """
        Master email template. Inline CSS only (Gmail-safe).

        Args:
            header_color: hex color for header band
            icon: emoji for header
            title: h2 title text
            subtitle: small text under title (usually timestamp)
            rows: concatenated _data_row() strings
            notice: optional info/warning box HTML
            footer_note: optional footer text
        """
        ts = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
        if not subtitle:
            subtitle = ts

        notice_html = ""
        if notice:
            notice_html = (
                f'<div style="margin:16px 24px;padding:12px 16px;'
                f'background:#FEF3C7;border-left:4px solid #D97706;'
                f'border-radius:4px;font-family:Arial,Helvetica,sans-serif;'
                f'font-size:13px;color:#92400E;">{notice}</div>'
            )

        footer_extra = ""
        if footer_note:
            footer_extra = (
                f'<p style="margin:0 0 8px;font-family:Arial,Helvetica,'
                f'sans-serif;font-size:12px;color:#9CA3AF;">'
                f'{footer_note}</p>'
            )

        return f'''<div style="background:{BRAND['background']};padding:24px 0;margin:0;">
<table cellpadding="0" cellspacing="0" border="0" width="100%"
       style="max-width:600px;margin:0 auto;">

  <!-- HEADER -->
  <tr><td style="background:{header_color};padding:20px 24px;
      border-radius:8px 8px 0 0;">
    <span style="font-size:24px;">{icon}</span>
    <span style="font-family:Arial,Helvetica,sans-serif;font-size:18px;
          font-weight:700;color:#FFFFFF;margin-left:8px;
          letter-spacing:1px;">{BRAND['name']}</span>
    <span style="font-family:Arial,Helvetica,sans-serif;font-size:11px;
          color:rgba(255,255,255,0.7);display:block;margin-top:4px;
          letter-spacing:0.5px;">{BRAND['tagline']}</span>
  </td></tr>

  <!-- BODY -->
  <tr><td style="background:{BRAND['surface']};padding:0;">

    <!-- Title Block -->
    <div style="padding:24px 24px 16px;">
      <h2 style="margin:0;font-family:Arial,Helvetica,sans-serif;
          font-size:20px;color:#111827;">{title}</h2>
      <p style="margin:4px 0 0;font-family:Arial,Helvetica,sans-serif;
         font-size:12px;color:#9CA3AF;">{subtitle}</p>
    </div>

    <!-- Data Table -->
    <table cellpadding="0" cellspacing="0" border="0" width="100%">
      {rows}
    </table>

    {notice_html}

    <!-- CTA Button -->
    <div style="padding:20px 24px;text-align:center;">
      <a href="{self._dashboard_url}"
         style="display:inline-block;padding:12px 28px;
         background:{BRAND['accent']};color:#FFFFFF;
         font-family:Arial,Helvetica,sans-serif;font-size:14px;
         font-weight:600;text-decoration:none;border-radius:6px;">
        Open Dashboard →
      </a>
    </div>

  </td></tr>

  <!-- FOOTER -->
  <tr><td style="padding:16px 24px;text-align:center;
      border-top:1px solid #E5E7EB;">
    {footer_extra}
    <p style="margin:0;font-family:Arial,Helvetica,sans-serif;
       font-size:11px;color:#D1D5DB;">
      This is an automated message from {BRAND['name']}. Do not reply.
    </p>
  </td></tr>

</table>
</div>'''

    # ─────────────────────────────────────────────────────────────────
    # PUBLIC — 7 Canonical Alert Methods
    # ─────────────────────────────────────────────────────────────────

    def kill_switch_fired(
        self,
        tier: str,
        drawdown_pct: float,
        portfolio_value: float,
        positions_closed: int = 0,
    ) -> None:
        """🔴 CRITICAL — Kill switch triggered."""
        try:
            if not self._enabled:
                logger.debug("Alerting disabled — skipping")
                return

            subject = f"🔴 APEX ALGO — Kill Switch Tier {tier} Activated"
            rows = "".join([
                self._data_row("Kill Switch Tier", tier,
                               BRAND["danger"]),
                self._data_row("Current Drawdown",
                               f"{drawdown_pct:.1%}",
                               BRAND["danger"]),
                self._data_row("Portfolio Value",
                               f"€{portfolio_value:,.2f}"),
                self._data_row("Positions Closed",
                               str(positions_closed)),
            ])
            notice = (
                "⚠️ <strong>All new entries have been halted.</strong> "
                "Manual review required before resuming trading."
            )
            html = self._base_template(
                BRAND["danger"], "🔴", "Kill Switch Activated",
                f"Tier {tier} · {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}",
                rows, notice=notice,
                footer_note=f"Triggered at {drawdown_pct:.1%} drawdown",
            )
            self._send_email(subject, html, "CRITICAL")

        except Exception as e:
            logger.error(f"Alert failed silently: {e}")

    def daily_summary(
        self,
        portfolio_value: float,
        daily_pnl: float,
        daily_pnl_pct: float = 0.0,
        open_positions: int = 0,
        regime: str = "UNKNOWN",
        trades_today: int = 0,
        new_lessons: int = 0,
        ytd_return: float = 0.0,
        sharpe_30d: float = 0.0,
    ) -> None:
        """📊 NORMAL — End-of-day daily report."""
        try:
            if not self._enabled:
                logger.debug("Alerting disabled — skipping")
                return

            pnl_emoji = "🟢" if daily_pnl >= 0 else "🔴"
            pnl_color = BRAND["success"] if daily_pnl >= 0 else BRAND["danger"]
            today_str = date.today().strftime("%Y-%m-%d")

            subject = (
                f"📊 APEX ALGO — Daily Report {today_str} {pnl_emoji}"
            )

            regime_colors = {
                "TRENDING": BRAND["success"],
                "RANGING": BRAND["warning"],
                "RISK_OFF": BRAND["danger"],
            }

            rows = "".join([
                self._data_row("Portfolio Value",
                               f"€{portfolio_value:,.2f}"),
                self._data_row("Daily P&L",
                               f"€{daily_pnl:+,.2f} ({daily_pnl_pct:+.2f}%)",
                               pnl_color),
                self._data_row("YTD Return",
                               f"{ytd_return:+.2f}%",
                               BRAND["success"] if ytd_return >= 0 else BRAND["danger"]),
                self._data_row("Sharpe (30d)",
                               f"{sharpe_30d:.3f}"),
                self._data_row("Open Positions", str(open_positions)),
                self._data_row("Trades Today", str(trades_today)),
                self._data_row("Market Regime", regime,
                               regime_colors.get(regime, BRAND["neutral"])),
                self._data_row("New Lessons", str(new_lessons),
                               BRAND["intelligence"] if new_lessons > 0 else "#111827"),
            ])

            html = self._base_template(
                BRAND["primary"], "📊", f"Daily Report · {today_str}",
                f"Market close · {pnl_emoji} {daily_pnl_pct:+.2f}%",
                rows,
            )
            self._send_email(subject, html, "NORMAL")

        except Exception as e:
            logger.error(f"Alert failed silently: {e}")

    def trade_executed(
        self,
        symbol: str,
        side: str,
        shares: float,
        price: float,
        dollar_value: float = 0.0,
        strategy: str = "",
        signal_strength: float = 0.0,
        regime: str = "UNKNOWN",
        portfolio_value_after: float = 0.0,
    ) -> None:
        """📈/📉 NORMAL — Trade execution notification."""
        try:
            if not self._enabled:
                logger.debug("Alerting disabled — skipping")
                return

            is_buy = side.upper() == "BUY"
            emoji = "📈" if is_buy else "📉"
            header_color = BRAND["success"] if is_buy else BRAND["accent"]
            value = dollar_value or (shares * price)

            subject = (
                f"{emoji} APEX ALGO — {side.upper()} {symbol} · "
                f"€{value:,.0f}"
            )

            rows = "".join([
                self._data_row("Symbol", symbol),
                self._data_row("Side", side.upper(),
                               BRAND["success"] if is_buy else BRAND["accent"]),
                self._data_row("Shares", f"{shares:.4f}"),
                self._data_row("Price", f"€{price:,.2f}"),
                self._data_row("Value", f"€{value:,.2f}"),
                self._data_row("Strategy", strategy),
                self._data_row("Signal Strength",
                               f"{signal_strength:.3f}"),
                self._data_row("Market Regime", regime),
                self._data_row("Portfolio After",
                               f"€{portfolio_value_after:,.2f}")
                if portfolio_value_after else "",
            ])

            html = self._base_template(
                header_color, emoji,
                f"{side.upper()} {symbol}",
                f"€{value:,.2f} · {strategy}",
                rows,
            )
            self._send_email(subject, html, "NORMAL")

        except Exception as e:
            logger.error(f"Alert failed silently: {e}")

    def api_failure(self, service: str, error: str) -> None:
        """⚠️ CRITICAL — External API failure."""
        try:
            if not self._enabled:
                logger.debug("Alerting disabled — skipping")
                return

            subject = f"⚠️ APEX ALGO — API Failure: {service}"

            rows = "".join([
                self._data_row("Service", service, BRAND["warning"]),
                self._data_row("Error", str(error)[:200],
                               BRAND["danger"]),
                self._data_row("Time",
                               datetime.now(timezone.utc).strftime(
                                   "%Y-%m-%d %H:%M:%S UTC")),
            ])
            notice = (
                "The trading system may be operating with degraded "
                "functionality. Check the dashboard for current status."
            )

            html = self._base_template(
                BRAND["warning"], "⚠️", f"API Failure: {service}",
                "", rows, notice=notice,
            )
            self._send_email(subject, html, "CRITICAL")

        except Exception as e:
            logger.error(f"Alert failed silently: {e}")

    def system_started(self, mode: str,
                       portfolio_value: float = 0.0) -> None:
        """🚀 NORMAL — System boot notification."""
        try:
            if not self._enabled:
                logger.debug("Alerting disabled — skipping")
                return

            subject = f"🚀 APEX ALGO — System Started ({mode})"

            rows = "".join([
                self._data_row("Mode", mode.upper(),
                               BRAND["success"] if mode.lower() == "paper"
                               else BRAND["danger"]),
                self._data_row("Portfolio Value",
                               f"€{portfolio_value:,.2f}"),
                self._data_row("Started At",
                               datetime.now(timezone.utc).strftime(
                                   "%Y-%m-%d %H:%M:%S UTC")),
            ])

            html = self._base_template(
                BRAND["primary"], "🚀",
                f"System Started — {mode.upper()} Mode", "", rows,
            )
            self._send_email(subject, html, "NORMAL")

        except Exception as e:
            logger.error(f"Alert failed silently: {e}")

    def system_stopped(self, reason: str = "Unknown") -> None:
        """⛔ CRITICAL — System stopped unexpectedly."""
        try:
            if not self._enabled:
                logger.debug("Alerting disabled — skipping")
                return

            subject = "⛔ APEX ALGO — System Stopped"

            rows = "".join([
                self._data_row("Reason", reason, BRAND["danger"]),
                self._data_row("Stopped At",
                               datetime.now(timezone.utc).strftime(
                                   "%Y-%m-%d %H:%M:%S UTC")),
            ])

            notice = (
                "⚠️ <strong>The trading system has stopped.</strong> "
                "Manual restart required. Check logs for details."
            )

            html = self._base_template(
                BRAND["neutral"], "⛔", "System Stopped",
                "", rows, notice=notice,
            )
            self._send_email(subject, html, "CRITICAL")

        except Exception as e:
            logger.error(f"Alert failed silently: {e}")

    def test_alerts(self) -> dict:
        """✅ NORMAL — Send test email and return result."""
        try:
            subject = "✅ APEX ALGO — Alert Test · Configured Correctly"

            rows = "".join([
                self._data_row("Status", "All Systems Operational",
                               BRAND["success"]),
                self._data_row("SMTP Host",
                               getattr(self.cfg, "SMTP_HOST", "N/A")),
                self._data_row("Recipient",
                               getattr(self.cfg, "ALERT_EMAIL", "N/A")),
                self._data_row("CC",
                               getattr(self.cfg, "ALERT_EMAIL_CC", "")
                               or "None"),
                self._data_row("Sent At",
                               datetime.now(timezone.utc).strftime(
                                   "%Y-%m-%d %H:%M:%S UTC")),
            ])

            html = self._base_template(
                BRAND["primary"], "✅",
                "Alert System Test", "Configuration verified", rows,
                footer_note="This is a test email. If you received it, "
                            "your alert configuration is correct.",
            )

            sent = self._send_email(subject, html, "NORMAL")
            return {"email_sent": sent, "email_error": None}

        except Exception as e:
            logger.error(f"Alert failed silently: {e}")
            return {"email_sent": False, "email_error": str(e)}
