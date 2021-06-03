=======
Tasks
=======

Here is the code for the `luigi <https://github.com/spotify/luigi/>`_ tasks that support the custom management command load sample data from `Random Data API <https://random-data-api.com/>`_. A few highlights, patterns worth repeating:

* :code:`DjangoModelTarget` counts the number of rows inserted into the database as a way of establishing whether a load task has been completed
* :code:`BaseLoadTask` inherits from :code:`ForceableTask` which will overwrite any existing fixture files. This carries all the way back from the final :code:`AllTablesTask`
* :code:`parent_dir` parameter makes it so that fixture creation tasks and table loading tasks are fully testable in a temporary directory and mock database.

::

    class DjangoModelTarget(Target):
        """Django model target that tests whether a model has the specified number of rows."""

        def __init__(self, count: int, model: Model):
            self.count = count
            self.model = model

        def get(self):
            return self.model.objects.all().count()

        def exists(self):
            try:
                count = self.get()
                if count == self.count:
                    return True
                else:
                    return False
            except self.model.DoesNotExist:
                return False


    class BaseLoadTask(ForceableTask):
        """Abstract base class to avoid repeating parent_dir, start and end date
        boilerplate.
        """

        parent_dir = Parameter(default="grants/fixtures")
        sample_size = IntParameter(default=100)
        requires = Requires()

        def run(self):
            NotImplementedError()


    class CompanyFixtureTask(BaseLoadTask):
        """Creates fixture for Company."""

        filename = Parameter(default="company.jsonl")

        def output(self):
            return LocalTarget(f"{self.parent_dir}/{self.filename}")

        def run(self):
            base_url = "https://random-data-api.com/api/"
            end_point_company = "company/random_company"
            end_point_address = "address/random_address"

            r = requests.get(
                f"{base_url}{end_point_company}", params=dict(size=self.sample_size)
            )

            df_company = pd.DataFrame(r.json())

            r = requests.get(
                f"{base_url}{end_point_address}", params=dict(size=self.sample_size)
            )

            df_address = pd.DataFrame(r.json())

            df = pd.concat(
                [df_company[["business_name"]], df_address[["city", "state"]]], axis=1
            )

            df.columns = ["name", "city", "state"]
            df["last_modified"] = datetime.datetime.now().isoformat()
            df["created_date"] = datetime.datetime.now().isoformat()

            with self.output().open("w") as f:
                for k, row in df.iterrows():
                    record = {"pk": k + 1, "model": "grants.Company", "fields": dict(row)}
                    f.write(json.dumps(record) + "\n")


    class CompanyTableTask(BaseLoadTask):
        """Loads data to company table."""

        fixture = Requirement(CompanyFixtureTask)

        def run(self):
            call_command(loaddata.Command(), self.input()["fixture"].path)

        def output(self):
            return DjangoModelTarget(model=Company, count=self.sample_size)


    class ApplicationFixtureTask(BaseLoadTask):
        """Creates fixture for Application."""

        company_table = Requirement(CompanyTableTask)
        filename = Parameter(default="application.jsonl")

        def output(self):
            return LocalTarget(f"{self.parent_dir}/{self.filename}")

        def run(self):
            with self.output().open("w") as f:
                for c in self.input()["company_table"].model.objects.all():
                    app_record = dict(
                        pk=c.id,
                        model="grants.Application",
                        fields=dict(
                            company=c.id,
                            created_date=c.created_date.isoformat(),
                            last_modified=c.last_modified.isoformat(),
                            status=100,
                        ),
                    )
                    f.write(json.dumps(app_record) + "\n")


    class ApplicationTableTask(BaseLoadTask):
        """Loads data to Application table."""

        fixture = Requirement(ApplicationFixtureTask)

        def run(self):
            call_command(loaddata.Command(), self.input()["fixture"].path)

        def output(self):
            return DjangoModelTarget(model=Application, count=self.sample_size)


    class ReviewerFixtureTask(BaseLoadTask):
        """Creates fixture for Reviewer."""

        filename = Parameter(default="reviewer.jsonl")

        def output(self):
            return LocalTarget(f"{self.parent_dir}/{self.filename}")

        def run(self):
            base_url = "https://random-data-api.com/api/"
            end_point = "users/random_user"

            r = requests.get(f"{base_url}{end_point}", params=dict(size=self.sample_size))

            df = pd.DataFrame(r.json())[["email", "last_name", "first_name"]]

            df["last_modified"] = datetime.datetime.now().isoformat()
            df["created_date"] = datetime.datetime.now().isoformat()
            df["status"] = 100

            with self.output().open("w") as f:
                for k, row in df.iterrows():
                    record = {
                        "pk": k + 1,
                        "model": "grants.Reviewer",
                        "fields": dict(row),
                    }
                    f.write(json.dumps(record) + "\n")


    class ReviewerTableTask(BaseLoadTask):
        """Loads data to Reviewer table."""

        fixture = Requirement(ReviewerFixtureTask)

        def run(self):
            call_command(loaddata.Command(), self.input()["fixture"].path)

        def output(self):
            return DjangoModelTarget(model=Reviewer, count=self.sample_size)


    class AllTablesTask(BaseLoadTask):
        """
        Somewhat of a wrapper task to load all tables. Uses reviewer for output
        even though all the requirements have have their own output targets. Used
        this instead of Wrapper to be able to pass down properties from BaseLoadTask.
        """

        companies = Requirement(CompanyTableTask)
        applications = Requirement(ApplicationTableTask)
        reviewers = Requirement(ReviewerTableTask)
