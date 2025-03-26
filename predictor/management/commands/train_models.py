from django.core.management.base import BaseCommand
from predictor.models import train_models

class Command(BaseCommand):
    help = 'Trains all salary prediction models'

    def handle(self, *args, **options):
        self.stdout.write('Starting model training...')
        train_models()
        self.stdout.write(self.style.SUCCESS('Successfully trained all models')) 