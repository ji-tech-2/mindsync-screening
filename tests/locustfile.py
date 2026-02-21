from locust import HttpUser, task, between


class MindSyncUser(HttpUser):
    # Jeda 1 sampai 3 detik antar request
    wait_time = between(1, 3)

    # Base URL
    host = "https://api.mindsync.my"

    @task
    def create_prediction(self):
        # Pastikan untuk mengecek apakah API ini butuh token/API key di header
        # Jika tidak butuh otentikasi, kamu bisa menghapus "Authorization"
        headers = {"Content-Type": "application/json"}

        # Payload sesuai format yang kamu berikan
        # Perhatikan bahwa 'null' ditulis sebagai 'None' di Python
        payload = {
            "age": 25,
            "gender": "Male",
            "occupation": "Student",
            "work_mode": "Remote",
            "work_screen_hours": 6,
            "leisure_screen_hours": 2,
            "sleep_hours": 7,
            "sleep_quality_1_5": 4,
            "stress_level_0_10": 8,
            "productivity_0_100": 65,
            "exercise_minutes_per_week": 150,
            "social_hours_per_week": 10,
            "mental_wellness_index_0_100": None,
        }

        # Mengirim request dan memvalidasi status 200 OK
        with self.client.post(
            "/v1/predictions/create", json=payload, headers=headers, catch_response=True
        ) as response:
            if (
                response.status_code == 200
                or response.status_code == 201
                or response.status_code == 202
            ):
                response.success()
            else:
                # Jika gagal, tampilkan pesan error dari server untuk mempermudah proses debug
                response.failure(
                    f"Failed! Status: {response.status_code}, Text: {response.text}"
                )
