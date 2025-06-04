import * as cdk from 'aws-cdk-lib';
import { Construct } from 'constructs';
import * as iam from 'aws-cdk-lib/aws-iam';
import * as sagemaker from 'aws-cdk-lib/aws-sagemaker';

export class SagemakerNotebookStack extends cdk.Stack {
  constructor(scope: Construct, id: string, props?: cdk.StackProps) {
    super(scope, id, props);

    // IAM Role
    const SageMakerNotebookinstanceRole = new iam.Role(this, 'SageMakerNotebookInstanceRole', {
      assumedBy: new iam.ServicePrincipal('sagemaker.amazonaws.com'),
      managedPolicies: [
        iam.ManagedPolicy.fromAwsManagedPolicyName('AmazonBedrockFullAccess'),
        iam.ManagedPolicy.fromAwsManagedPolicyName('AmazonOpenSearchServiceFullAccess'),
        iam.ManagedPolicy.fromAwsManagedPolicyName('AmazonSageMakerFullAccess'),
        iam.ManagedPolicy.fromAwsManagedPolicyName('AmazonSSMFullAccess'),
        iam.ManagedPolicy.fromAwsManagedPolicyName('AmazonS3FullAccess'),
        iam.ManagedPolicy.fromAwsManagedPolicyName('SecretsManagerReadWrite')
      ],
    });
    
    // Create a lifecycle configuration to handle repository setup
    const lifecycleConfig = new sagemaker.CfnNotebookInstanceLifecycleConfig(this, 'FinopsNotebookLifecycleConfig', {
      notebookInstanceLifecycleConfigName: 'finops-notebook-lifecycle-config',
      onCreate: [
        {
          content: cdk.Fn.base64(`
            #!/bin/bash
            set -e
            # No need to clone the repository as it's handled by defaultCodeRepository
            # Just ensure proper permissions
            cd /home/ec2-user/SageMaker
            if [ -d "finops-demo" ]; then
              sudo chown -R ec2-user:ec2-user finops-demo
              echo "Repository permissions set successfully"
            fi
            `)
        }
      ],
      onStart: [
        {
          content: cdk.Fn.base64(`
            #!/bin/bash
            set -e
            cd /home/ec2-user/SageMaker/finops-demo
            git config --global --add safe.directory /home/ec2-user/SageMaker/finops-demo
            git pull
            echo "Repository updated successfully"
            `)
        }
      ]
    });
    
    // SageMaker Notebook Instance
    const cfnNotebookInstance = new sagemaker.CfnNotebookInstance(this, 'MyCfnNotebookInstance', {
      instanceType: 'ml.m5.xlarge',
      roleArn: SageMakerNotebookinstanceRole.roleArn,
      defaultCodeRepository: 'https://github.com/ottlseo/finops-demo.git',
      directInternetAccess: 'Enabled',
      notebookInstanceName: 'finops-notebook-instance',
      volumeSizeInGb: 50,
      lifecycleConfigName: lifecycleConfig.notebookInstanceLifecycleConfigName,
    });
  }
}
