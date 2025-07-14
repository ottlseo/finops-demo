import { Stack, StackProps, Fn } from 'aws-cdk-lib';
import { Construct } from 'constructs';
import * as cdk from 'aws-cdk-lib';
import * as ec2 from 'aws-cdk-lib/aws-ec2';
import * as iam from 'aws-cdk-lib/aws-iam';
import * as fs from 'fs';
import * as path from 'path';
import * as ssm from 'aws-cdk-lib/aws-ssm';
import * as secretsmanager from 'aws-cdk-lib/aws-secretsmanager';

export class EC2Stack extends Stack {
  constructor(scope: Construct, id: string, props?: StackProps) {
    super(scope, id, props);

    // IAM Role to access EC2
    const instanceRole = new iam.Role(this, 'InstanceRole', {
      assumedBy: new iam.ServicePrincipal('ec2.amazonaws.com'),
      managedPolicies: [
        iam.ManagedPolicy.fromAwsManagedPolicyName('AdministratorAccess'),
      ],
    });

    // Add permissions to access SSM Parameter Store and Secrets Manager
    instanceRole.addToPolicy(new iam.PolicyStatement({
      actions: [
        'ssm:GetParameter',
        'secretsmanager:GetSecretValue'
      ],
      resources: ['*'],
    }));

    // Network setting for EC2
    const defaultVpc = ec2.Vpc.fromLookup(this, 'VPC', {
      isDefault: true,
    });

    const chatbotAppSecurityGroup = new ec2.SecurityGroup(this, 'chatbotAppSecurityGroup', {
      vpc: defaultVpc,
    });
    chatbotAppSecurityGroup.addIngressRule(
      ec2.Peer.anyIpv4(),
      ec2.Port.tcp(80),
      'httpIpv4',
    );
    chatbotAppSecurityGroup.addIngressRule(
      ec2.Peer.anyIpv4(),
      ec2.Port.tcp(22),
      'sshIpv4',
    );
    chatbotAppSecurityGroup.addIngressRule(
      ec2.Peer.anyIpv4(),
      ec2.Port.tcp(8501),
      'streamlitIpv4',
    );

    // set AMI
    const machineImage = ec2.MachineImage.fromSsmParameter(
      '/aws/service/canonical/ubuntu/server/jammy/stable/current/amd64/hvm/ebs-gp2/ami-id'
    );
    
    // set User Data
    const userData = ec2.UserData.forLinux();
    const userDataScript = fs.readFileSync(path.join(__dirname, 'userdata.sh'), 'utf8');
    userData.addCommands(userDataScript);
    
    // Add env.sh script to create .env file with OpenSearch values
    const envScript = fs.readFileSync(path.join(__dirname, 'env.sh'), 'utf8');
    userData.addCommands(envScript);
    
    // EC2 instance
    const chatbotAppInstance = new ec2.Instance(this, 'chatbotAppInstance', {
      instanceType: new ec2.InstanceType('m5.large'),
      machineImage: machineImage,
      vpc: defaultVpc,
      securityGroup: chatbotAppSecurityGroup,
      role: instanceRole,
      userData: userData,
    });

    new cdk.CfnOutput(this, 'ChatbotAppUrl', {
      value: `http://${chatbotAppInstance.instancePublicIp}/`,
      description: 'The URL of AWS FinOps Chatbot instance - Please wait for 5 minutes from now',
      exportName: 'ChatbotAppUrl',
    });
  }
}
