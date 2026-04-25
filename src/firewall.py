import subprocess
import platform
import os


class Firewall:
    """
    Bloque et débloque des adresses IP
    via le pare-feu du système d'exploitation.
    Fonctionne sur Windows et Linux.
    """

    def __init__(self):
        self.os_type     = platform.system()
        self.blocked_ips = set()

    def block_ip(self, ip: str,
                 reason: str = "") -> bool:
        """
        Bloque une adresse IP.
        Retourne True si succès, False sinon.
        """
        if ip in self.blocked_ips:
            return True  # déjà bloquée

        # Ignorer les IPs locales
        if self._is_local(ip):
            return False

        success = False

        if self.os_type == "Windows":
            success = self._block_windows(ip)
        elif self.os_type == "Linux":
            success = self._block_linux(ip)
        else:
            print(
                f"OS non supporté : {self.os_type}"
            )
            return False

        if success:
            self.blocked_ips.add(ip)
            print(
                f"[FIREWALL] IP bloquée : {ip}"
                + (f" — {reason}" if reason else "")
            )
        return success

    def unblock_ip(self, ip: str) -> bool:
        """
        Débloque une adresse IP.
        """
        if ip not in self.blocked_ips:
            return True

        success = False
        if self.os_type == "Windows":
            success = self._unblock_windows(ip)
        elif self.os_type == "Linux":
            success = self._unblock_linux(ip)

        if success:
            self.blocked_ips.discard(ip)
            print(f"[FIREWALL] IP débloquée : {ip}")
        return success

    def unblock_all(self):
        """Débloque toutes les IPs bloquées."""
        for ip in list(self.blocked_ips):
            self.unblock_ip(ip)

    def _is_local(self, ip: str) -> bool:
        """Vérifie si l'IP est locale/privée."""
        local_prefixes = (
            '127.', '192.168.', '10.',
            '172.16.', '172.17.', '172.18.',
            '172.19.', '172.20.', '172.21.',
            '172.22.', '172.23.', '172.24.',
            '172.25.', '172.26.', '172.27.',
            '172.28.', '172.29.', '172.30.',
            '172.31.', '0.0.0.0', '255.'
        )
        return ip.startswith(local_prefixes)

    # ── Windows ────────────────────────────────
    def _block_windows(self, ip: str) -> bool:
        rule_name = f"BLOCK_ANOMALY_{ip}"
        cmd = [
            "netsh", "advfirewall", "firewall",
            "add", "rule",
            f"name={rule_name}",
            "dir=in",
            "action=block",
            f"remoteip={ip}",
            "enable=yes"
        ]
        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=10
            )
            return result.returncode == 0
        except Exception as e:
            print(f"[FIREWALL] Erreur Windows : {e}")
            return False

    def _unblock_windows(self, ip: str) -> bool:
        rule_name = f"BLOCK_ANOMALY_{ip}"
        cmd = [
            "netsh", "advfirewall", "firewall",
            "delete", "rule",
            f"name={rule_name}"
        ]
        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=10
            )
            return result.returncode == 0
        except Exception as e:
            print(f"[FIREWALL] Erreur Windows : {e}")
            return False

    # ── Linux ──────────────────────────────────
    def _block_linux(self, ip: str) -> bool:
        cmd = [
            "iptables", "-A", "INPUT",
            "-s", ip, "-j", "DROP"
        ]
        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=10
            )
            return result.returncode == 0
        except Exception as e:
            print(f"[FIREWALL] Erreur Linux : {e}")
            return False

    def _unblock_linux(self, ip: str) -> bool:
        cmd = [
            "iptables", "-D", "INPUT",
            "-s", ip, "-j", "DROP"
        ]
        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=10
            )
            return result.returncode == 0
        except Exception as e:
            print(f"[FIREWALL] Erreur Linux : {e}")
            return False