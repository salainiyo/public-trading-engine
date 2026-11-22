import unittest

from engine.models import Trade, ChallengeState, StrategyState


class TestTrade(unittest.TestCase):
    def test_creation(self):
        t = Trade(
            id="t1", strategy="CO", direction=1,
            entry=100.0, sl=99.0, tp=103.0,
            lots=0.1, open_time="2025-01-15T10:00:00",
        )
        self.assertEqual(t.id, "t1")
        self.assertEqual(t.direction, 1)
        self.assertEqual(t.status, "open")
        self.assertIsNone(t.close_time)
        self.assertIsNone(t.pnl)
        self.assertEqual(t.position_id, None)

    def test_defaults(self):
        t = Trade(
            id="t2", strategy="CO", direction=-1,
            entry=200.0, sl=201.0, tp=197.0,
            lots=0.5, open_time="2025-01-15T10:00:00",
        )
        self.assertEqual(t._hold_bars, 0)
        self.assertEqual(t._trail_px, 0.0)
        self.assertEqual(t._sl_dist, 0.0)


class TestChallengeState(unittest.TestCase):
    def test_creation(self):
        ch = ChallengeState(
            strategy="CO", challenge_num=1,
            equity=5000.0, balance=5000.0,
            phase_start_bal=5000.0, dd_floor=4500.0,
            day_equity=5000.0,
        )
        self.assertEqual(ch.phase, 1)
        self.assertFalse(ch.blown)
        self.assertEqual(ch.trading_days, 0)
        self.assertIsNone(ch.open_trade)
        self.assertEqual(ch.open_trades, [])

    def test_active_challenges(self):
        s = StrategyState(name="CO")
        ch1 = ChallengeState(strategy="CO", challenge_num=1,
                             equity=5000, balance=5000,
                             phase_start_bal=5000, dd_floor=4500,
                             day_equity=5000)
        ch2 = ChallengeState(strategy="CO", challenge_num=2,
                             equity=5000, balance=5000,
                             phase_start_bal=5000, dd_floor=4500,
                             day_equity=5000, blown=True)
        s.challenges = [ch1, ch2]
        self.assertEqual(len(s.active_challenges), 1)
        self.assertEqual(s.active_challenges[0].challenge_num, 1)

    def test_current_challenge_active(self):
        s = StrategyState(name="CO")
        ch1 = ChallengeState(strategy="CO", challenge_num=1,
                             equity=5000, balance=5000,
                             phase_start_bal=5000, dd_floor=4500,
                             day_equity=5000)
        s.challenges = [ch1]
        self.assertIs(s.current_challenge, ch1)

    def test_current_challenge_blown(self):
        s = StrategyState(name="CO")
        ch1 = ChallengeState(strategy="CO", challenge_num=1,
                             equity=5000, balance=5000,
                             phase_start_bal=5000, dd_floor=4500,
                             day_equity=5000, blown=True)
        ch2 = ChallengeState(strategy="CO", challenge_num=2,
                             equity=5000, balance=5000,
                             phase_start_bal=5000, dd_floor=4500,
                             day_equity=5000)
        s.challenges = [ch1, ch2]
        self.assertEqual(s.current_challenge.challenge_num, 2)


if __name__ == "__main__":
    unittest.main()
