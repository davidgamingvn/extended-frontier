resource "google_compute_network" "vpc_network" {
  name                    = "my-custom-mode-network"
  auto_create_subnetworks = false
  mtu                     = 1460
}

resource "google_compute_subnetwork" "default" {
  name          = "my-custom-subnet"
  ip_cidr_range = "10.0.1.0/24"
  region        = "us-west1"
  network       = google_compute_network.vpc_network.id
}

resource "google_compute_instance" "default" {
  name         = "frontiermap-backend"
  machine_type = "e2-medium"
  zone         = "us-west1-a"
  tags         = ["ssh", "http-server"]

  boot_disk {
    initialize_params {
      image = "debian-cloud/debian-11"
    }
  }

  # Install Flask
  metadata_startup_script = <<-EOT
    #!/bin/bash
    sudo apt-get update
    sudo apt-get install -yq build-essential python3-pip rsync
    pip install flask
  EOT

  network_interface {
    subnetwork = google_compute_subnetwork.default.id

    access_config {
      # Automatically assigns an ephemeral external IP address
    }
  }
}

resource "google_compute_firewall" "ssh" {
  name = "allow-ssh"
  allow {
    ports    = ["22"]
    protocol = "tcp"
  }
  direction     = "INGRESS"
  network       = google_compute_network.vpc_network.id
  priority      = 1000
  source_ranges = ["0.0.0.0/0"]
  target_tags   = ["ssh"]
}

# resource "google_compute_firewall" "flask" {
#   name    = "allow-flask"
#   network = google_compute_network.vpc_network.id

#   allow {
#     protocol = "tcp"
#     ports    = ["5000"]
#   }
#   direction     = "INGRESS"
#   source_ranges = ["0.0.0.0/0"]
#   priority      = 1000
#   target_tags   = ["http-server"]
# }

output "web_server_url" {
  value = join("", ["http://", google_compute_instance.default.network_interface.0.access_config.0.nat_ip, ":5000"])
}
